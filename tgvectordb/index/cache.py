from collections import OrderedDict
from typing import Optional

from tgvectordb.utils.config import DEFAULT_CACHE_MAX_ITEMS


class VectorCache:
    def __init__(self, max_items: int = None):
        self.max_items = max_items or DEFAULT_CACHE_MAX_ITEMS
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, message_id: int) -> Optional[dict]:
        if message_id in self._cache:
            self._cache.move_to_end(message_id)
            self._hits += 1
            return self._cache[message_id]
        self._misses += 1
        return None

    def put(self, message_id: int, data: dict):
        if message_id in self._cache:
            self._cache.move_to_end(message_id)
            self._cache[message_id] = data
            return
        if len(self._cache) >= self.max_items:
            self._cache.popitem(last=False)
        self._cache[message_id] = data

    def put_batch(self, items: dict):
        for message_id, data in items.items():
            self.put(message_id, data)

    def has(self, message_id: int) -> bool:
        return message_id in self._cache

    def get_many(self, message_ids: list) -> tuple:
        cached = {}
        uncached = []
        for message_id in message_ids:
            result = self.get(message_id)
            if result is not None:
                cached[message_id] = result
            else:
                uncached.append(message_id)
        return cached, uncached

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_items,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }
