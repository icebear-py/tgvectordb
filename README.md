# TgVectorDB

**Free, unlimited vector database backed by Telegram.**

Store embeddings as Telegram messages. Search them semantically. No API keys. No servers. No monthly bills. Just your Telegram account.

> *Turbopuffer for broke developers.*

## how it works

Your vectors are stored as messages in a private Telegram channel you own. A tiny local index (~1MB) routes queries to the right cluster. Search fetches only the relevant messages — not the whole database.

```
cold query:  ~0.5-1.5 seconds  (fetching from telegram)
warm query:  <5 milliseconds   (from local cache)
cost:        $0/month forever
```

## install

```bash
pip install tgvectordb

# for pdf support:
pip install tgvectordb[pdf]
```

## get telegram credentials

1. go to https://my.telegram.org
2. log in with your phone number
3. click "API development tools"
4. create an app, grab the `api_id` and `api_hash`

## quick start

```python
from tgvectordb import TgVectorDB

db = TgVectorDB(
    api_id=12345,
    api_hash="your_api_hash_here",
    phone="+91xxxxxxxxxx",
    db_name="my-notes",
)

# add some text
db.add("photosynthesis converts sunlight into chemical energy in plants")
db.add("neural networks learn patterns from training data")
db.add("sourdough bread requires a long fermentation process")

# search
results = db.search("how do plants make food?", top_k=3)
for r in results:
    print(f"[{r['score']:.2f}] {r['text'][:80]}...")

# add a whole document
db.add_source("research_paper.pdf")

# check stats
print(db.stats())
```

## use with a RAG chatbot

```python
from tgvectordb import TgVectorDB

db = TgVectorDB(api_id=..., api_hash=..., phone=..., db_name="rag-bot")

# index your docs (one time)
db.add_source("handbook.pdf")
db.add_source("faq.md")

# on each question
def answer(question):
    context = db.search(question, top_k=5)
    context_text = "\n".join([r["text"] for r in context])
    # pass to your LLM of choice (ollama, llama.cpp, openai, whatever)
    return ask_llm(f"Context:\n{context_text}\n\nQuestion: {question}")
```

## features

- **free forever** — telegram provides unlimited cloud storage at no cost
- **zero infrastructure** — no docker, no servers, no databases to manage
- **durable** — your data lives on telegram's multi-datacenter infrastructure
- **portable** — `db.restore()` on any new machine and you're back in business
- **fast enough** — 0.5-1.5s cold queries, <5ms warm queries with caching
- **private** — data stays in your own private telegram channel

## important stuff

- uses `intfloat/e5-small-v2` for embeddings (384 dims, runs on CPU)
- vectors are int8 quantized to fit in telegram's 4096 char message limit
- uses Telethon (MTProto) for fast message fetching, not the bot API
- recommended: use a secondary telegram account, not your main one
- this is for personal projects and prototyping, not production SaaS

## commands

```python
# --- Adding Data ---
db.add("text")                          
# Adds a single string of text to the database.

db.add_batch(texts, metadatas)          
# Adds multiple texts at once with optional metadata. Much faster than calling add() in a loop since it batches the embeddings and Telegram messages.

db.add_source("file.pdf")              
# Automatically parses, chunks, and adds a supported file format (.pdf, .docx, .txt, .md, .html, .csv, .json, .py, etc).

db.add_directory("./my_docs/", extensions=[".pdf", ".docx"])  
# Ingests all supported files within a directory concurrently.

# --- Search & Retrieval ---
db.search("query", top_k=5)            
# Performs a semantic similarity search against the stored vectors and returns the top_k most relevant chunks.

db.search("query", filter={"src": "document.pdf"}) 
# Filters the semantic search to ONLY check against vectors matching that exact metadata tag.

# --- Database Management ---
db.reindex()                           
# Forces a full re-clustering of your entire vector database. Normally this is handled automatically as your dataset grows.

db.backup()                            
# Zips up the small local SQLite index map and uploads it to a private Telegram channel for disaster recovery.

db.restore()                           
# Downloads the latest index backup from Telegram, allowing you to instantly restore your database on any machine.

db.stats()                             
# Returns a dictionary of database statistics, including vector counts, cache hit rates, and clustering metrics.

db.delete(filter={"src": "old.pdf"})   
# Permanently removes all vectors and metadata matching a specific filter from both Telegram and the local index.
```

## supported file formats

works out of the box: `.txt`, `.md`, `.html`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.xml`, `.yaml`, `.py`, `.js`, `.java`, `.go`, `.rs`, and most text-based files.

with optional dependencies:
- `.pdf` — `pip install tgvectordb[pdf]` (uses pdfplumber)
- `.docx` — `pip install tgvectordb[docx]` (uses python-docx)
- or just: `pip install tgvectordb[all]`

## license

MIT — do whatever you want with it.

## disclaimer

this project uses telegram's cloud infrastructure as a storage backend. while projects like Pentaract have done this since 2023 without issues, its not officially sanctioned by telegram. use a secondary account and don't abuse rate limits. see the full disclaimer in the docs.
