import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from tgvectordb.embedding.model import EmbeddingModel
from tgvectordb.embedding.quantizer import Quantizer
from tgvectordb.index.cache import VectorCache
from tgvectordb.index.clustering import (
    assign_to_nearest_cluster,
    compute_num_clusters,
    find_nearest_clusters,
    run_kmeans,
)
from tgvectordb.index.store import LocalIndex
from tgvectordb.ingestors.registry import ingest, is_supported
from tgvectordb.search.engine import rank_results
from tgvectordb.telegram.connection import TelegramConnection
from tgvectordb.telegram.messages import (
    delete_messages,
    download_latest_file,
    fetch_messages_by_ids,
    send_vector_messages_batch,
    upload_file_to_channel,
)
from tgvectordb.utils.config import (
    CLUSTERING_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_NPROBE,
    REINDEX_GROWTH_TRIGGER,
)
from tgvectordb.utils.serialization import pack_vector_message

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
    _start_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, _bg_loop)
    return future.result()


class TgVectorDB:
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
        self._tg = TelegramConnection(api_id, api_hash, phone, db_name, data_path)
        self._model = EmbeddingModel(model_name or DEFAULT_MODEL_NAME)
        self._index = LocalIndex(db_name, data_path)
        self._cache = VectorCache(cache_max_items)
        self._quantizer = None
        self._vectors_channel = None
        self._index_channel = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        _run(self._async_init())

    async def _async_init(self):
        await self._tg.connect()
        self._vectors_channel = await self._tg.get_or_create_channel("vectors")
        self._index_channel = await self._tg.get_or_create_channel("index")
        dims = self._model.get_dimensions()
        self._quantizer = Quantizer(dims)
        saved_model = self._index.get_config("model_name")
        if saved_model is None:
            self._index.set_config("model_name", self._model.model_name)
            self._index.set_config("dimensions", dims)
            self._index.set_config("channel_id", str(self._vectors_channel.id))
        else:
            if saved_model != self._model.model_name:
                raise ValueError(
                    f"this database was created with model '{saved_model}' "
                    f"but you're trying to use '{self._model.model_name}'. "
                    f"you cant mix models - embeddings would be incompatible."
                )
        self._initialized = True

    def add(self, text: str, metadata: dict = None):
        self._ensure_init()
        metadata = metadata or {}
        vector = self._model.embed_document(text)
        int8_vector, quantization_parameters = self._quantizer.quantize(vector)
        message_string = pack_vector_message(
            int8_vector, quantization_parameters, metadata, text=text
        )
        channel_id_str = str(self._vectors_channel.id)
        message_id = _run(
            send_vector_messages_batch(
                self._tg.get_client(),
                self._vectors_channel,
                [message_string],
            )
        )[0]
        centroids = self._index.load_centroids(self._quantizer.dims)
        if centroids is not None:
            cluster_id = assign_to_nearest_cluster(vector, centroids)
        else:
            cluster_id = 0
        self._index.add_to_cluster(cluster_id, message_id, channel_id_str)
        self._maybe_reindex()

    def add_batch(self, texts: list, metadatas: list = None):
        self._ensure_init()
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        print(f"embedding {len(texts)} chunks...")
        vectors = self._model.embed_documents_batch(texts)
        int8_vectors, qparams_list = self._quantizer.quantize_batch(vectors)
        message_strings = []
        for i in range(len(texts)):
            message_string = pack_vector_message(
                int8_vectors[i], qparams_list[i], metadatas[i], text=texts[i]
            )
            message_strings.append(message_string)
        print(f"sending {len(message_strings)} messages to telegram...")

        def on_progress(done, total):
            print(f"  sent {done}/{total}")

        message_ids = _run(
            send_vector_messages_batch(
                self._tg.get_client(),
                self._vectors_channel,
                message_strings,
                progress_callback=on_progress,
            )
        )
        centroids = self._index.load_centroids(self._quantizer.dims)
        channel_id_str = str(self._vectors_channel.id)
        entries = []
        for i, message_id in enumerate(message_ids):
            if centroids is not None:
                cluster_id = assign_to_nearest_cluster(vectors[i], centroids)
            else:
                cluster_id = 0
            entries.append((cluster_id, message_id, channel_id_str))
        self._index.add_to_cluster_batch(entries)
        print(f"added {len(message_ids)} vectors to database")
        self._maybe_reindex()

    def add_source(self, filepath: str, chunk_size: int = 400, overlap: int = 50):
        from pathlib import Path as P

        filename = P(filepath).name
        if not is_supported(filepath):
            raise ValueError(
                f"dont know how to read '{P(filepath).suffix}' files. "
                f"try: .pdf, .docx, .txt, .md, .html, .csv, .json"
            )
        print(f"ingesting: {filename}")
        chunks = ingest(filepath, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print(f"no text found in {filename}!")
            return
        print(f"  extracted {len(chunks)} chunks from {filename}")
        texts = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {"src": c.get("src", filename), "chunk_idx": c.get("chunk_idx", 0)}
            if "format" in c:
                meta["format"] = c["format"]
            metadatas.append(meta)
        self.add_batch(texts, metadatas)

    def add_directory(
        self,
        dirpath: str,
        extensions: list = None,
        recursive: bool = True,
        chunk_size: int = 400,
        overlap: int = 50,
    ):
        from pathlib import Path as P

        directory_path = P(dirpath)
        if not directory_path.is_dir():
            raise ValueError(f"not a directory: {dirpath}")
        if recursive:
            all_files = list(directory_path.rglob("*"))
        else:
            all_files = list(directory_path.glob("*"))
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
        for i, filepath in enumerate(sorted(files_to_add)):
            print(f"\n[{i + 1}/{len(files_to_add)}] ", end="")
            try:
                self.add_source(str(filepath), chunk_size=chunk_size, overlap=overlap)
            except Exception as e:
                print(f"  error with {filepath.name}: {e}")
                continue
        print("\ndirectory ingestion complete")

    def search(self, query: str, top_k: int = 5, filter: dict = None) -> list:
        self._ensure_init()
        total_vecs = self._index.get_total_vectors()
        if total_vecs == 0:
            return []
        query_vec = self._model.embed_query(query)
        centroids = self._index.load_centroids(self._quantizer.dims)
        if centroids is None or total_vecs < CLUSTERING_THRESHOLD:
            all_msg_data = self._index.get_all_message_ids()
            target_msg_ids = [message_id for message_id, _ in all_msg_data]
        else:
            nearest = find_nearest_clusters(query_vec, centroids, self.nprobe)
            target_msg_ids = []
            for cluster_id, _score in nearest:
                cluster_msgs = self._index.get_cluster_message_ids(cluster_id)
                for message_id, _ in cluster_msgs:
                    target_msg_ids.append(message_id)
        if not target_msg_ids:
            return []
        cached, uncached_ids = self._cache.get_many(target_msg_ids)
        if uncached_ids:
            fetched = _run(
                fetch_messages_by_ids(
                    self._tg.get_client(),
                    self._vectors_channel,
                    uncached_ids,
                )
            )
            self._cache.put_batch(fetched)
            cached.update(fetched)
        filter_fn = None
        if filter:

            def filter_fn(meta):
                for key, val in filter.items():
                    if meta.get(key) != val:
                        return False
                return True

        results = rank_results(
            query_vec, cached, self._quantizer, top_k=top_k, filter_fn=filter_fn
        )
        return results

    def reindex(self):
        self._ensure_init()
        self._do_reindex()

    def backup(self):
        self._ensure_init()
        db_path = str(self._index.get_db_path())
        print("backing up index to telegram...")
        message_id = _run(
            upload_file_to_channel(
                self._tg.get_client(),
                self._index_channel,
                db_path,
                caption=f"TgVectorDB index backup - {self.db_name}",
            )
        )
        print(f"index backed up (msg_id: {message_id})")

    def restore(self):
        self._ensure_init()
        save_path = str(self._index.get_db_path())
        self._index.close()
        print("downloading index from telegram...")
        found = _run(
            download_latest_file(
                self._tg.get_client(),
                self._index_channel,
                save_path,
            )
        )
        if found:
            self._index = LocalIndex(self.db_name)
            print("index restored successfully!")
        else:
            print("no backup found in the index channel")

    def stats(self) -> dict:
        self._ensure_init()
        total = self._index.get_total_vectors()
        n_clusters = self._index.get_num_clusters()
        cache_stats = self._cache.stats()
        last_backup = self._index.get_config("last_backup_count", 0)
        last_reindex = self._index.get_config("last_reindex_count", 0)
        if total < CLUSTERING_THRESHOLD:
            search_mode = f"flat (brute-force all {total} vectors, clustering kicks in at {CLUSTERING_THRESHOLD})"
        else:
            search_mode = (
                f"IVF clustered ({n_clusters} clusters, probing top {self.nprobe})"
            )
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
        self._ensure_init()
        all_msgs = self._index.get_all_message_ids()
        cached, uncached = self._cache.get_many([m for m, _ in all_msgs])
        sources = set()
        for data in cached.values():
            source = data.get("metadata", {}).get("src", "")
            if source:
                sources.add(source)
        return sorted(sources)

    def delete(self, filter: dict = None):
        self._ensure_init()
        if not filter:
            raise ValueError("please provide a filter. deleting everything is scary.")
        all_msgs = self._index.get_all_message_ids()
        all_msg_ids = [m for m, _ in all_msgs]
        all_data = _run(
            fetch_messages_by_ids(
                self._tg.get_client(),
                self._vectors_channel,
                all_msg_ids,
            )
        )
        to_delete = []
        for message_id, data in all_data.items():
            meta = data.get("metadata", {})
            match = all(meta.get(k) == v for k, v in filter.items())
            if match:
                to_delete.append(message_id)
        if not to_delete:
            print("no matching vectors found")
            return
        print(f"deleting {len(to_delete)} vectors...")
        _run(delete_messages(self._tg.get_client(), self._vectors_channel, to_delete))
        self._index.delete_by_message_ids(to_delete)
        for message_id in to_delete:
            if self._cache.has(message_id):
                pass
        print(f"deleted {len(to_delete)} vectors")

    def _maybe_reindex(self):
        total = self._index.get_total_vectors()
        last_reindex = self._index.get_config("last_reindex_count", 0)
        last_backup = self._index.get_config("last_backup_count", 0)
        vectors_since_backup = total - last_backup
        should_backup = vectors_since_backup >= 10 or (last_backup == 0 and total > 0)
        if total < CLUSTERING_THRESHOLD:
            if should_backup:
                self._do_backup()
            return
        if last_reindex == 0:
            self._do_reindex()
            return
        growth = (total - last_reindex) / max(last_reindex, 1)
        if growth >= REINDEX_GROWTH_TRIGGER:
            self._do_reindex()
        elif should_backup:
            self._do_backup()

    def _do_backup(self):
        total = self._index.get_total_vectors()
        self.backup()
        self._index.set_config("last_backup_count", total)

    def _do_reindex(self):
        total = self._index.get_total_vectors()
        if total < CLUSTERING_THRESHOLD:
            print(f"only {total} vectors, too few for clustering. skipping reindex.")
            return
        print(f"reindexing {total} vectors...")
        start = time.time()
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
        message_ids_ordered = []
        vectors_float = []
        for message_id in all_msg_ids:
            if message_id not in cached:
                continue
            data = cached[message_id]
            float_vec = self._quantizer.dequantize(
                data["vector_int8"], data["quant_params"]
            )
            message_ids_ordered.append(message_id)
            vectors_float.append(float_vec)
        if not vectors_float:
            print("no vectors found to cluster!")
            return
        vectors_array = np.array(vectors_float, dtype=np.float32)
        n_clusters = compute_num_clusters(len(vectors_array))
        centroids = run_kmeans(vectors_array, n_clusters)
        self._index.clear_cluster_map()
        self._index.save_centroids(centroids)
        entries = []
        for i, message_id in enumerate(message_ids_ordered):
            cluster_id = assign_to_nearest_cluster(vectors_float[i], centroids)
            channel_id = all_channel_ids.get(message_id, str(self._vectors_channel.id))
            entries.append((cluster_id, message_id, channel_id))
        self._index.add_to_cluster_batch(entries)
        self._index.set_config("last_reindex_count", len(message_ids_ordered))
        self._index.set_config("last_backup_count", len(message_ids_ordered))
        elapsed = time.time() - start
        print(
            f"reindex complete: {len(message_ids_ordered)} vectors → {n_clusters} clusters ({elapsed:.1f}s)"
        )
        self.backup()

    def close(self):
        self._index.close()
        _run(self._tg.disconnect())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        total = self._index.get_total_vectors() if self._initialized else "?"
        return f"<TgVectorDB '{self.db_name}' vectors={total}>"
