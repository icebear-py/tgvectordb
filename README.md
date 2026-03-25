# TgVectorDB

**The free, unlimited cloud vector database backed by... Telegram.**

Store embeddings directly as Telegram messages. Search them semantically. No API keys. No Docker. No vector index scaling limits. No monthly VC-subsidized cloud bills. Just your Telegram account.

> *Turbopuffer for broke CS students surviving on instant ramen.*

## How it works

Your vectors are stored as messages in a private Telegram channel you own. A tiny local index (~1MB) routes queries to the correct cluster locally, and then we fetch only the strictly relevant messages via MTProto. We don't download the whole database.

Why? Because I am building a personal chatbot and refuse to pay $70/mo for a managed enterprise vector database to store my notes PDF collections that i wont even need in next semester.

```yaml
Cold Query:  ~0.5 - 2.0 seconds (asking Telegram for messages)
Warm Query:  < 5 milliseconds (it's cached locally now)
Cost:        $0/month forever (perfect for your non-existent startup budget)
Scalability: Unlimited (or until Parel durov notices you are storing your entire university library)
```

## Installation

TgVectorDB comes with built-in capabilities to ingest PDFs, DOCX, and text embeddings via Telegram directly. One command handles it all:

```bash
pip install tgvectordb
```

## Getting Started

### 1. Get Telegram Credentials

1. Go to https://my.telegram.org
2. Log in with your phone number.
3. Click on "API development tools".
4. Create an application and copy your `api_id` and `api_hash`.

### 2. Quick Start

```python
from tgvectordb import TgVectorDB

db = TgVectorDB(
    api_id=12345,
    api_hash="your_api_hash_here",
    phone="+91xxxxxxxxxx",
    db_name="my-notes",
)

# Add single texts
db.add("Photosynthesis converts sunlight into chemical energy in plants, which I need to know for tomorrow's exam.")
db.add("Neural networks learn patterns from training data. I just copy code from stackoverflow.")

# Search semantically
results = db.search("How do I pass biology without studying?", top_k=3)
for result in results:
    print(f"[{result['score']:.2f}] {result['text'][:80]}...")

# Ingest an entire document
db.add_source("CS101_final_cheatsheet_vFINAL_v2.pdf")

# Print database stats
print(db.stats())
```

## Building a RAG Chatbot (The Real Reason You Are Here to Automate Homework)

Building Retrieval-Augmented Generation (RAG) is entirely free here. Perfect for hobbyists, tinkerers, and broke students trying to build a personal AI tutor at 3 AM the night before an assignment is due.

```python
from tgvectordb import TgVectorDB

db = TgVectorDB(
    api_id=12345, 
    api_hash="your_api_hash", 
    phone="+91xxxxxxxxxx", 
    db_name="last-minute-homework-bot"
)

# Toss your chaotic life knowledge and class slides into the void (One-time setup)
db.add_source("professor_ramble_transcript.pdf")
db.add_source("assignment_that_makes_no_sense.md")

# Query context on the fly
def answer(question: str):
    context = db.search(question, top_k=5)
    context_text = "\n".join([r["text"] for r in context])
    
    # Pass to your local LLM (Ollama, vLLM, etc.)
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    return ask_llm(prompt)
```

## Features

- **Free Forever:** Telegram provides unlimited cloud storage entirely at no cost. Thanks, Telegram.
- **Zero Infrastructure:** No Docker containers, servers, or external databases to manage.
- **Highly Durable:** Your data safely resides on Telegram's multi-datacenter infrastructure.
- **Fully Portable:** Run `db.backup()` on one machine and `db.restore()` on another. You're fully back up and running.
- **Fast Search:** 0.5-1.5s for cold queries, <5ms for warm queries with our extremely complex intelligent caching.
- **Private & Secure:** Your data stays within your private Telegram channels.

## Architecture & Details

- Uses `intfloat/e5-small-v2` for embeddings (384 dimensions, runs perfectly on CPU).
- Vectors are `int8` quantized to fit strictly within Telegram's 4096-character message limits.
- Powered by `Telethon` (MTProto) for high-speed message fetching directly from the network, bypassing normal Bot API restrictions.
- **Strong Recommendation:** Use a secondary, dedicated Telegram account instead of your primary personal account. If you get rate limited, you don't want your main chats delayed.

## API Reference

### Database Operations
```python
db.add("text")                          # Add single text passage
db.add_batch(texts, metadatas)          # Add multiple texts optimally
db.add_source("file.pdf")               # Parse and add a PDF file
db.add_source("notes.docx")             # Parse and add a Word document
db.add_source("data.csv")               # Add a CSV (auto-converts to semantic text)
db.add_source("code.py")                # Add a raw code file
db.add_directory("./my_docs/")          # Recursively add all supported files from a folder
db.add_directory("./docs", extensions=[".pdf", ".docx"])  # Filter directory ingestion

# Search & Retrieval
db.search("query", top_k=5)             # Perform semantic search
db.search("query", filter={"src": "x"}) # Search combined with metadata filtering

# Maintenance
db.reindex()                            # Force dataset re-clustering for IVF
db.backup()                             # Push local index mapping over to Telegram
db.restore()                            # Restore local index mapping from Telegram
db.delete(filter={"src": "old.pdf"})    # Delete specific vectors matching a rule
db.stats()                              # Display database telemetry
```

## Supported Formats

All these formats are seamlessly extracted with a basic `pip install tgvectordb`, requiring no messy external boilerplate!

`.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.xml`, `.yaml`, `.py`, `.js`, `.java`, `.go`, `.rs`, and most text-based source code files.

## Disclaimer

This library is a **hobbyist experimental project** designed for side-projects, panic-built student chatbots, and folks who don't want to pay Qdrant or any other cloud vectordb provider when their bank account has $4 in it. It is practically a satire of modern VC-backed enterprise vector databases. It works genuinely well, but please do not run your mission-critical, HIPAA-compliant enterprise SaaS on top of my Telegram hack. Because if it breaks, the only customer support you're getting is me reading your GitHub issue and closing it.

**Note:** This project ingeniously (or stupidly) leverages Telegram's cloud infrastructure as a backend storage. While projects like *Pentaract* have achieved this since 2023 with excellent success, this is not an officially promoted enterprise use-case by Telegram. Please use a secondary account and respectfully avoid abusing rate limits so we don't ruin this for all the other broke students.

## License

**MIT License** — Do whatever you want with it!