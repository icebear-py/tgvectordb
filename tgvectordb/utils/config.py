"""
all the constants and defaults live here.
change stuff here if you know what you're doing.
"""

import os
from pathlib import Path


# where we keep local data (index, session files, etc)
DEFAULT_DATA_DIR = Path.home() / ".tgvectordb"

# embedding model - e5-small is the sweet spot for us
# 384 dims, fast as hell, good quality, fits nicely in telegram messages
DEFAULT_MODEL_NAME = "intfloat/e5-small-v2"
DEFAULT_DIMENSIONS = 384

# telegram message limit is 4096 chars
# int8 vector at 384 dims = 512 base64 chars
# leaves us ~3500 chars for metadata + text chunk
TELEGRAM_MSG_CHAR_LIMIT = 4096
MAX_TEXT_CHUNK_CHARS = 2500  # safe limit for the text field in metadata

# how many messages we can fetch in one telegram api call
TG_FETCH_BATCH_SIZE = 100

# rate limit stuff - dont wanna get banned lol
TG_MAX_SEND_PER_SEC = 25  # bit under 30 to be safe
TG_SEND_DELAY = 1.0 / TG_MAX_SEND_PER_SEC
TG_MAX_FETCH_PER_SEC = 28  # same idea, stay under the limit

# when to switch from flat search (brute force everything) to clustered
# below this number, clustering is pointless overhead
CLUSTERING_THRESHOLD = 1000

# how many clusters to create - auto calculated but these are bounds
MIN_CLUSTERS = 8
MAX_CLUSTERS = 128

# reindex when this much new data has been added (as fraction of total)
REINDEX_GROWTH_TRIGGER = 0.10  # 10%

# how many closest clusters to check during search
# more = better recall but slower (more telegram fetches)
DEFAULT_NPROBE = 3

# messages per channel before we consider making an overflow channel
CHANNEL_OVERFLOW_THRESHOLD = 700_000

# LRU cache defaults
DEFAULT_CACHE_MAX_ITEMS = 50_000  # vectors cached in memory

# channel naming
CHANNEL_PREFIX = "tgvdb"
