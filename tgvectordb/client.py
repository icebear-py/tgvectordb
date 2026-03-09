"""
TgVectorDB - the main class that users interact with.

this is the only thing most people need to import:
    from tgvectordb import TgVectorDB

it handles:
  - connecting to telegram
  - embedding text
  - storing vectors as messages
  - searching with IVF routing
  - reindexing when data grows
  - backing up the index
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np

from tgvectordb.telegram.connection import TelegramConnection
from tgvectordb.telegram.messages import (
    send_vector_messages_batch,
    fetch_messages_by_ids,
    fetch_all_messages,
    delete_messages,
    upload_file_to_channel,
    download_latest_file,
)
from tgvectordb.embedding.model import EmbeddingModel
from tgvectordb.embedding.quantizer import Quantizer
from tgvectordb.embedding.chunker import chunk_text
from tgvectordb.ingestors.registry import ingest, is_supported
from tgvectordb.index.store import LocalIndex
from tgvectordb.index.clustering import (
    compute_num_clusters,
    run_kmeans,
    assign_to_nearest_cluster,
    find_nearest_clusters,
)
from tgvectordb.index.cache import VectorCache
from tgvectordb.search.engine import rank_results
from tgvectordb.utils.config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_NPROBE,
    CLUSTERING_THRESHOLD,
    REINDEX_GROWTH_TRIGGER,
    CHANNEL_OVERFLOW_THRESHOLD,
)
from tgvectordb.utils.serialization import pack_vector_message


import threading

_bg_loop = None
_bg_thread = None

def _start_bg_loop():
    global _bg_loop, _bg_thread
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(_bg_loop)
            _bg_loop.run_forever()
        _bg_thread = threading.Thread(target=run_loop, daemon=True)
        _bg_thread.start()

def _run(coro):
    """helper to run async code from sync context. uses a persistent background loop."""
    _start_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return future.result()


class TgVectorDB:
    """
    your vector database, backed by telegram.

    basic usage:
        db = TgVectorDB(api_id=123, api_hash="abc", phone="+91xxx")
        db.add("some text about machine learning")
        results = db.search("what is deep learning?", top_k=5)
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        phone: str,
        db_name: str = "default",
        model_name: str = None,
        nprobe: int = None,
        cache_max_items: int = None,
        data_dir: str = None,
    ):
        self.db_name = db_name
        self.nprobe = nprobe or DEFAULT_NPROBE

        data_path = Path(data_dir) if data_dir else None

        # set up all the components
        self._tg = TelegramConnection(api_id, api_hash, phone, db_name, data_path)
        self._model = EmbeddingModel(model_name or DEFAULT_MODEL_NAME)
        self._index = LocalIndex(db_name, data_path)
        self._cache = VectorCache(cache_max_items)
        self._quantizer = None  # created after we know dimensions

        # telegram channels - set up on first use
        self._vectors_channel = None
        self._index_channel = None
        self._initialized = False

    def _ensure_init(self):
        """lazy initialization - connects to telegram on first real operation."""
        if self._initialized:
            return
        _run(self._async_init())

    async def _async_init(self):
        """the actual init - async because telegram needs it."""
        await self._tg.connect()

        # get or create our channels
        self._vectors_channel = await self._tg.get_or_create_channel("vectors")
        self._index_channel = await self._tg.get_or_create_channel("index")

        # figure out dimensions from model
        dims = self._model.get_dimensions()
        self._quantizer = Quantizer(dims)

        # save config if first time, check if existing
        saved_model = self._index.get_config("model_name")
        if saved_model is None:
            # fresh database
            self._index.set_config("model_name", self._model.model_name)
            self._index.set_config("dimensions", dims)
            self._index.set_config("channel_id", str(self._vectors_channel.id))
        else:
            # existing database - make sure model matches
            if saved_model != self._model.model_name:
                raise ValueError(
                    f"this database was created with model '{saved_model}' "
                    f"but you're trying to use '{self._model.model_name}'. "
                    f"you cant mix models - embeddings would be incompatible."
                )

        self._initialized = True

    # ---- PUBLIC API: Adding data ----

    def add(self, text: str, metadata: dict = None):
        """
        add a single text to the database.
        embeds it, quantizes, sends to telegram, updates index.
        """
        self._ensure_init()
        metadata = metadata or {}

        vec = self._model.embed_document(text)
        int8_vec, qparams = self._quantizer.quantize(vec)

        msg_str = pack_vector_message(int8_vec, qparams, metadata, text=text)

        # send to telegram
        channel_id_str = str(self._vectors_channel.id)
        msg_id = _run(
            send_vector_messages_batch(
                self._tg.get_client(),
                self._vectors_channel,
                [msg_str],
            )
        )[0]

        # figure out which cluster this belongs to
        centroids = self._index.load_centroids(self._quantizer.dims)
        if centroids is not None:
            cluster_id = assign_to_nearest_cluster(vec, centroids)
        else:
            cluster_id = 0  # no clusters yet, everything in cluster 0

        self._index.add_to_cluster(cluster_id, msg_id, channel_id_str)

        # check if we should reindex
        self._maybe_reindex()

    def add_batch(self, texts: list, metadatas: list = None):
        """
        add multiple texts at once. faster than calling add() in a loop
        because it batches the embedding and telegram sends.
        """
        self._ensure_init()

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")

        print(f"embedding {len(texts)} chunks...")
        vectors = self._model.embed_documents_batch(texts)
        int8_vecs, qparams_list = self._quantizer.quantize_batch(vectors)

        # build message strings
        msg_strings = []
        for i in range(len(texts)):
            msg_str = pack_vector_message(
                int8_vecs[i], qparams_list[i], metadatas[i], text=texts[i]
            )
            msg_strings.append(msg_str)

        # send all to telegram
        print(f"sending {len(msg_strings)} messages to telegram...")

        def on_progress(done, total):
            print(f"  sent {done}/{total}")

        msg_ids = _run(
            send_vector_messages_batch(
                self._tg.get_client(),
                self._vectors_channel,
                msg_strings,
                progress_callback=on_progress,
            )
        )

        # assign to clusters
        centroids = self._index.load_centroids(self._quantizer.dims)
        channel_id_str = str(self._vectors_channel.id)

        entries = []
        for i, msg_id in enumerate(msg_ids):
            if centroids is not None:
                cid = assign_to_nearest_cluster(vectors[i], centroids)
            else:
                cid = 0
            entries.append((cid, msg_id, channel_id_str))

        self._index.add_to_cluster_batch(entries)
        print(f"added {len(msg_ids)} vectors to database")

        self._maybe_reindex()

    def add_source(self, filepath: str, chunk_size: int = 400, overlap: int = 50):
        """
        add a document file to the database.
        automatically detects format, extracts text, chunks it, and adds everything.

        supported formats:
            .pdf, .docx, .txt, .md, .html, .csv, .json, .jsonl,
            .py, .js, and most other text-based files.

        for pdf: pip install pdfplumber (or pip install tgvectordb[pdf])
        for docx: pip install python-docx (or pip install tgvectordb[docx])
        """
        from pathlib import Path as P
        fname = P(filepath).name

        if not is_supported(filepath):
            raise ValueError(
                f"dont know how to read '{P(filepath).suffix}' files. "
                f"try: .pdf, .docx, .txt, .md, .html, .csv, .json"
            )

        print(f"ingesting: {fname}")
        chunks = ingest(filepath, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            print(f"no text found in {fname}!")
            return

        print(f"  extracted {len(chunks)} chunks from {fname}")

        texts = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {"src": c.get("src", fname), "chunk_idx": c.get("chunk_idx", 0)}
            if "format" in c:
                meta["format"] = c["format"]
            metadatas.append(meta)

        self.add_batch(texts, metadatas)

    def add_directory(self, dirpath: str, extensions: list = None, recursive: bool = True,
                      chunk_size: int = 400, overlap: int = 50):
        """
        add all supported files from a directory.

        args:
            dirpath: path to directory
            extensions: only include these extensions, e.g. [".pdf", ".txt"]
                       if None, includes all supported formats
            recursive: whether to look in subdirectories too
            chunk_size: words per chunk
            overlap: word overlap between chunks
        """
        from pathlib import Path as P
        dirp = P(dirpath)

        if not dirp.is_dir():
            raise ValueError(f"not a directory: {dirpath}")

        # collect all files
        if recursive:
            all_files = list(dirp.rglob("*"))
        else:
            all_files = list(dirp.glob("*"))

        # filter to supported files
        files_to_add = []
        for f in all_files:
            if not f.is_file():
                continue
            if extensions and f.suffix.lower() not in extensions:
                continue
            if is_supported(str(f)):
                files_to_add.append(f)

        if not files_to_add:
            print(f"no supported files found in {dirpath}")
            return

        print(f"found {len(files_to_add)} files to ingest")

        for i, fpath in enumerate(sorted(files_to_add)):
            print(f"\n[{i+1}/{len(files_to_add)}] ", end="")
            try:
                self.add_source(str(fpath), chunk_size=chunk_size, overlap=overlap)
            except Exception as e:
                print(f"  error with {fpath.name}: {e}")
                # dont let one bad file kill the whole batch
                continue

        print(f"\ndirectory ingestion complete")

    # ---- PUBLIC API: Searching ----

    def search(self, query: str, top_k: int = 5, filter: dict = None) -> list:
        """
        search for similar text in the database.

        args:
            query: what you're searching for
            top_k: how many results to return
            filter: optional metadata filter like {"src": "paper.pdf"}

        returns:
            list of dicts with text, score, metadata, message_id
        """
        self._ensure_init()

        total_vecs = self._index.get_total_vectors()
        if total_vecs == 0:
            return []

        # embed the query
        query_vec = self._model.embed_query(query)

        # figure out which messages to fetch
        centroids = self._index.load_centroids(self._quantizer.dims)

        if centroids is None or total_vecs < CLUSTERING_THRESHOLD:
            # flat search - fetch everything (small db)
            all_msg_data = self._index.get_all_message_ids()
            target_msg_ids = [mid for mid, _ in all_msg_data]
        else:
            # IVF search - only fetch relevant clusters
            nearest = find_nearest_clusters(query_vec, centroids, self.nprobe)
            target_msg_ids = []
            for cluster_id, _score in nearest:
                cluster_msgs = self._index.get_cluster_message_ids(cluster_id)
                for msg_id, _ in cluster_msgs:
                    target_msg_ids.append(msg_id)

        if not target_msg_ids:
            return []

        # check cache first, only fetch whats missing
        cached, uncached_ids = self._cache.get_many(target_msg_ids)

        if uncached_ids:
            # fetch from telegram
            fetched = _run(
                fetch_messages_by_ids(
                    self._tg.get_client(),
                    self._vectors_channel,
                    uncached_ids,
                )
            )
            # put in cache for next time
            self._cache.put_batch(fetched)
            # merge with cached results
            cached.update(fetched)

        # build filter function if needed
        filter_fn = None
        if filter:
            def filter_fn(meta):
                for key, val in filter.items():
                    if meta.get(key) != val:
                        return False
                return True

        # rank and return
        results = rank_results(
            query_vec, cached, self._quantizer, top_k=top_k, filter_fn=filter_fn
        )

        return results

    # ---- PUBLIC API: Management ----

    def reindex(self):
        """
        force a full reindex. re-clusters all vectors.
        normally this happens automatically but you can trigger it.
        """
        self._ensure_init()
        self._do_reindex()

    def backup(self):
        """upload the local index to telegram for disaster recovery."""
        self._ensure_init()

        db_path = str(self._index.get_db_path())
        print(f"backing up index to telegram...")

        msg_id = _run(
            upload_file_to_channel(
                self._tg.get_client(),
                self._index_channel,
                db_path,
                caption=f"TgVectorDB index backup - {self.db_name}",
            )
        )
        print(f"index backed up (msg_id: {msg_id})")

    def restore(self):
        """download the latest index backup from telegram."""
        self._ensure_init()

        save_path = str(self._index.get_db_path())
        self._index.close()  # close current db before overwriting

        print("downloading index from telegram...")
        found = _run(
            download_latest_file(
                self._tg.get_client(),
                self._index_channel,
                save_path,
            )
        )

        if found:
            # reopen the index with the restored data
            self._index = LocalIndex(self.db_name)
            print("index restored successfully!")
        else:
            print("no backup found in the index channel")

    def stats(self) -> dict:
        """get some useful info about the database."""
        self._ensure_init()

        total = self._index.get_total_vectors()
        n_clusters = self._index.get_num_clusters()
        cache_stats = self._cache.stats()
        last_backup = self._index.get_config("last_backup_count", 0)
        last_reindex = self._index.get_config("last_reindex_count", 0)

        if total < CLUSTERING_THRESHOLD:
            search_mode = f"flat (brute-force all {total} vectors, clustering kicks in at {CLUSTERING_THRESHOLD})"
        else:
            search_mode = f"IVF clustered ({n_clusters} clusters, probing top {self.nprobe})"

        return {
            "total_vectors": total,
            "search_mode": search_mode,
            "num_clusters": n_clusters,
            "model": self._model.model_name,
            "dimensions": self._quantizer.dims,
            "db_name": self.db_name,
            "index_path": str(self._index.get_db_path()),
            "last_reindex_at": last_reindex,
            "last_backup_at": last_backup,
            "vectors_since_backup": total - last_backup,
            "cache": cache_stats,
        }

    def list_sources(self) -> list:
        """list unique source files that have been added."""
        self._ensure_init()

        # kinda hacky but we need to scan messages for this
        # for now just look at what we have cached + index metadata
        # TODO: maintain a sources table in the index
        all_msgs = self._index.get_all_message_ids()
        cached, uncached = self._cache.get_many([m for m, _ in all_msgs])

        sources = set()
        for data in cached.values():
            src = data.get("metadata", {}).get("src", "")
            if src:
                sources.add(src)

        return sorted(sources)

    def delete(self, filter: dict = None):
        """
        delete vectors matching a filter.
        currently only supports filtering by metadata fields.

        example: db.delete(filter={"src": "old_document.pdf"})
        """
        self._ensure_init()

        if not filter:
            raise ValueError("please provide a filter. deleting everything is scary.")

        # we need to scan through messages to find matches
        # this is slow but deletion should be rare
        all_msgs = self._index.get_all_message_ids()
        all_msg_ids = [m for m, _ in all_msgs]

        # fetch all to check metadata
        all_data = _run(
            fetch_messages_by_ids(
                self._tg.get_client(),
                self._vectors_channel,
                all_msg_ids,
            )
        )

        to_delete = []
        for msg_id, data in all_data.items():
            meta = data.get("metadata", {})
            match = all(meta.get(k) == v for k, v in filter.items())
            if match:
                to_delete.append(msg_id)

        if not to_delete:
            print("no matching vectors found")
            return

        print(f"deleting {len(to_delete)} vectors...")

        # delete from telegram
        _run(delete_messages(self._tg.get_client(), self._vectors_channel, to_delete))

        # remove from local index
        self._index.delete_by_message_ids(to_delete)

        # clear those from cache too
        for mid in to_delete:
            if self._cache.has(mid):
                # cache doesnt have a remove method but we can just leave it
                # itll get evicted naturally
                pass

        print(f"deleted {len(to_delete)} vectors")

    # ---- INTERNAL: Reindexing ----

    def _maybe_reindex(self):
        """check if we should trigger a reindex based on growth."""
        total = self._index.get_total_vectors()
        last_reindex = self._index.get_config("last_reindex_count", 0)
        last_backup = self._index.get_config("last_backup_count", 0)

        # always do a backup if enough new vectors were added
        # even without clustering, the index (message id mappings) matters
        vectors_since_backup = total - last_backup
        should_backup = vectors_since_backup >= 10 or (last_backup == 0 and total > 0)

        if total < CLUSTERING_THRESHOLD:
            # too small for clustering but still backup the index
            if should_backup:
                self._do_backup()
            return

        if last_reindex == 0:
            # never been indexed and we have enough data now
            self._do_reindex()
            return

        growth = (total - last_reindex) / max(last_reindex, 1)
        if growth >= REINDEX_GROWTH_TRIGGER:
            self._do_reindex()
        elif should_backup:
            # not enough growth for reindex but enough for a backup
            self._do_backup()

    def _do_backup(self):
        """backup the local index to telegram. happens even without clustering."""
        total = self._index.get_total_vectors()
        self.backup()
        self._index.set_config("last_backup_count", total)

    def _do_reindex(self):
        """the actual reindex operation. fetches all vectors, re-clusters, rebuilds index."""
        total = self._index.get_total_vectors()
        if total < CLUSTERING_THRESHOLD:
            print(f"only {total} vectors, too few for clustering. skipping reindex.")
            return

        print(f"reindexing {total} vectors...")
        start = time.time()

        # fetch all vectors from telegram
        # first try using cache for what we have
        all_msg_data = self._index.get_all_message_ids()
        all_msg_ids = [m for m, _ in all_msg_data]
        all_channel_ids = {m: c for m, c in all_msg_data}

        cached, uncached = self._cache.get_many(all_msg_ids)

        if uncached:
            print(f"  fetching {len(uncached)} uncached vectors from telegram...")
            fetched = _run(
                fetch_messages_by_ids(
                    self._tg.get_client(),
                    self._vectors_channel,
                    uncached,
                )
            )
            self._cache.put_batch(fetched)
            cached.update(fetched)

        # dequantize all vectors to float32 for clustering
        msg_ids_ordered = []
        vectors_float = []
        for mid in all_msg_ids:
            if mid not in cached:
                continue  # message might have been deleted
            data = cached[mid]
            float_vec = self._quantizer.dequantize(
                data["vector_int8"], data["quant_params"]
            )
            msg_ids_ordered.append(mid)
            vectors_float.append(float_vec)

        if not vectors_float:
            print("no vectors found to cluster!")
            return

        vectors_array = np.array(vectors_float, dtype=np.float32)

        # decide cluster count and run kmeans
        n_clusters = compute_num_clusters(len(vectors_array))
        centroids = run_kmeans(vectors_array, n_clusters)

        # rebuild the cluster map
        self._index.clear_cluster_map()
        self._index.save_centroids(centroids)

        entries = []
        for i, mid in enumerate(msg_ids_ordered):
            cid = assign_to_nearest_cluster(vectors_float[i], centroids)
            channel_id = all_channel_ids.get(mid, str(self._vectors_channel.id))
            entries.append((cid, mid, channel_id))

        self._index.add_to_cluster_batch(entries)
        self._index.set_config("last_reindex_count", len(msg_ids_ordered))
        self._index.set_config("last_backup_count", len(msg_ids_ordered))

        elapsed = time.time() - start
        print(f"reindex complete: {len(msg_ids_ordered)} vectors → {n_clusters} clusters ({elapsed:.1f}s)")

        # auto-backup after reindex
        self.backup()

    def close(self):
        """clean up connections."""
        self._index.close()
        _run(self._tg.disconnect())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        total = self._index.get_total_vectors() if self._initialized else "?"
        return f"<TgVectorDB '{self.db_name}' vectors={total}>"
