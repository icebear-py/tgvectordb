"""
simple in-memory LRU cache for vectors we've fetched from telegram.

after the first query about "cooking", all those vectors stay in memory.
next query about "baking" hits the same cluster -> instant, no telegram fetch.

nothing fancy here, just a dict with eviction.
"""

from collections import OrderedDict
from typing import Optional

from tgvectordb.utils.config import DEFAULT_CACHE_MAX_ITEMS


class VectorCache:
    """
    caches parsed vector data by message_id.
    uses OrderedDict for O(1) LRU eviction.
    """

    def __init__(self, max_items: int = None):
        self.max_items = max_items or DEFAULT_CACHE_MAX_ITEMS
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, message_id: int) -> Optional[dict]:
        """
        look up a cached vector by message id.
        returns None if not cached.
        """
        if message_id in self._cache:
            # move to end (most recently used)
            self._cache.move_to_end(message_id)
            self._hits += 1
            return self._cache[message_id]

        self._misses += 1
        return None

    def put(self, message_id: int, data: dict):
        """
        cache a vector + metadata. evicts oldest if full.
        data should have keys: vector_int8, quant_params, metadata
        """
        if message_id in self._cache:
            self._cache.move_to_end(message_id)
            self._cache[message_id] = data
            return

        if len(self._cache) >= self.max_items:
            # pop oldest item
            self._cache.popitem(last=False)

        self._cache[message_id] = data

    def put_batch(self, items: dict):
        """cache multiple items at once. items = {msg_id: data, ...}"""
        for msg_id, data in items.items():
            self.put(msg_id, data)

    def has(self, message_id: int) -> bool:
        return message_id in self._cache

    def get_many(self, message_ids: list) -> tuple:
        """
        check cache for a list of message ids.
        returns (cached_dict, uncached_ids_list)
        """
        cached = {}
        uncached = []

        for mid in message_ids:
            result = self.get(mid)
            if result is not None:
                cached[mid] = result
            else:
                uncached.append(mid)

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
