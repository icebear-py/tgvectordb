"""
the local index is a small sqlite db that stores:
  - cluster centroids (for routing queries)
  - which message ids belong to which cluster
  - config stuff (model name, dimensions, etc)

this file should never be bigger than a few MB.
the actual vectors live on telegram - this is just the map.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional

from tgvectordb.utils.config import DEFAULT_DATA_DIR


class LocalIndex:
    """
    sqlite-backed index that maps clusters to telegram message ids.
    """

    def __init__(self, db_name: str, data_dir: Path = None):
        self.db_name = db_name
        self.data_dir = (data_dir or DEFAULT_DATA_DIR) / db_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "index.db"
        self._conn = None
        self._setup_db()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # faster for our use case
        return self._conn

    def _setup_db(self):
        """create tables if they dont exist yet."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS centroids (
                cluster_id INTEGER PRIMARY KEY,
                centroid BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cluster_map (
                message_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                channel_id TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cluster_map_cluster
                ON cluster_map(cluster_id);

            CREATE INDEX IF NOT EXISTS idx_cluster_map_msg
                ON cluster_map(message_id);
        """)
        conn.commit()

    # --- config stuff ---

    def get_config(self, key: str, default=None):
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def set_config(self, key: str, value):
        conn = self._get_conn()
        val_json = json.dumps(value)
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, val_json),
        )
        conn.commit()

    # --- centroid operations ---

    def save_centroids(self, centroids: np.ndarray):
        """
        save all cluster centroids. replaces any existing ones.
        centroids shape: (num_clusters, dims)
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM centroids")

        for i in range(centroids.shape[0]):
            blob = centroids[i].astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO centroids (cluster_id, centroid) VALUES (?, ?)",
                (i, blob),
            )
        conn.commit()

    def load_centroids(self, dims: int) -> Optional[np.ndarray]:
        """load all centroids. returns None if no centroids stored yet."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT cluster_id, centroid FROM centroids ORDER BY cluster_id"
        ).fetchall()

        if not rows:
            return None

        num_clusters = len(rows)
        centroids = np.zeros((num_clusters, dims), dtype=np.float32)

        for cluster_id, blob in rows:
            centroids[cluster_id] = np.frombuffer(blob, dtype=np.float32)

        return centroids

    def get_num_clusters(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM centroids").fetchone()
        return row[0]

    # --- cluster map operations ---

    def add_to_cluster(self, cluster_id: int, message_id: int, channel_id: str):
        """record that a message belongs to a cluster."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO cluster_map (message_id, cluster_id, channel_id) VALUES (?, ?, ?)",
            (message_id, cluster_id, channel_id),
        )
        conn.commit()

    def add_to_cluster_batch(self, entries: list):
        """
        bulk insert. entries = [(cluster_id, message_id, channel_id), ...]
        much faster than one by one.
        """
        conn = self._get_conn()
        conn.executemany(
            "INSERT INTO cluster_map (cluster_id, message_id, channel_id) VALUES (?, ?, ?)",
            entries,
        )
        conn.commit()

    def get_cluster_message_ids(self, cluster_id: int) -> list:
        """get all message ids + channel ids for a cluster."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT message_id, channel_id FROM cluster_map WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchall()
        return rows  # list of (msg_id, channel_id) tuples

    def get_total_vectors(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM cluster_map").fetchone()
        return row[0]

    def clear_cluster_map(self):
        """wipe the cluster map. used during reindex."""
        conn = self._get_conn()
        conn.execute("DELETE FROM cluster_map")
        conn.commit()

    def delete_by_message_ids(self, msg_ids: list):
        """remove specific message entries. for when vectors are deleted."""
        if not msg_ids:
            return
        conn = self._get_conn()
        placeholders = ",".join(["?"] * len(msg_ids))
        conn.execute(
            f"DELETE FROM cluster_map WHERE message_id IN ({placeholders})",
            msg_ids,
        )
        conn.commit()

    # --- convenience ---

    def get_all_message_ids(self) -> list:
        """get every message id we know about. for reindexing."""
        conn = self._get_conn()
        rows = conn.execute("SELECT message_id, channel_id FROM cluster_map").fetchall()
        return rows

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_db_path(self) -> Path:
        return self.db_path
