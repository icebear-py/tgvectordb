from tgvectordb import TgVectorDB


# ── your credentials ──
API_ID = 00000000
API_HASH = ""
PHONE = "+91xxxxxxxxxx"


def main():
    # ================================================================
    #  1. CONNECT TO DATABASE
    # ================================================================
    # creates channels on first run, reuses them after that.
    # first time will ask for telegram login code.

    db = TgVectorDB(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        db_name="quickstart-demo",
        # optional params:
        # nprobe=3,              # how many clusters to search (more = better recall, slower)
        # cache_max_items=50000, # how many vectors to keep in memory cache
        # data_dir="./my_data",  # custom path for local index (default: ~/.tgvectordb/)
    )

    # ================================================================
    #  2. ADDING DATA — single text
    # ================================================================
    # each add() call = 1 embedding + 1 telegram message

    db.add("The mitochondria is the powerhouse of the cell")
    db.add(
         "Python was created by Guido van Rossum in 1991",
         metadata={"topic": "programming", "src": "manual_entry"}
     )

    # ================================================================
    #  3. ADDING DATA — batch of texts
    # ================================================================
    # faster than calling add() in a loop because it batches
    # the embedding and telegram sends together

    # db.add_batch(
    #     texts=[
    #         "React is a JavaScript library for building user interfaces",
    #         "Docker containers provide lightweight OS-level virtualization",
    #         "PostgreSQL is an advanced open source relational database",
    #     ],
    #     metadatas=[
    #         {"topic": "frontend", "src": "tech_notes"},
    #         {"topic": "devops", "src": "tech_notes"},
    #         {"topic": "databases", "src": "tech_notes"},
    #     ]
    # )

    # ================================================================
    #  4. ADDING DATA — from files
    # ================================================================
    # auto-detects format, extracts text, chunks it, embeds, sends to telegram.
    # supported: .pdf .docx .txt .md .html .csv .json .jsonl .py .js etc

    # db.add_source("research_paper.pdf")
    # db.add_source("meeting_notes.docx")
    # db.add_source("readme.md")
    # db.add_source("employees.csv")
    # db.add_source("app.py")
    # db.add_source("blog_posts.jsonl")

    # with custom chunking (smaller chunks = more precise retrieval):
    # db.add_source("long_book.pdf", chunk_size=200, overlap=30)

    # ================================================================
    #  5. ADDING DATA — entire directory
    # ================================================================
    # recursively finds all supported files and ingests them

    # db.add_directory("./my_documents/")
    # db.add_directory("./research/", extensions=[".pdf", ".docx"])  # only these types
    # db.add_directory("./notes/", recursive=False)                  # top level only

    # ================================================================
    #  6. SEARCHING — basic semantic search
    # ================================================================
    print("\n--- basic search ---")
    results = db.search("how does machine learning work?", top_k=5)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:100]}...")
        print(f"          source: {r['metadata'].get('src', 'unknown')}")
        print()

    # ================================================================
    #  7. SEARCHING — with metadata filter
    # ================================================================
    # only search within vectors from a specific source or topic

    # results = db.search("revenue growth", filter={"src": "report.pdf"})
    # results = db.search("python tips", filter={"topic": "programming"})

    # ================================================================
    #  8. SEARCHING — adjusting result count
    # ================================================================
    # top_k controls how many results come back (default 5)

    # results = db.search("neural networks", top_k=10)   # more results
    # results = db.search("specific fact", top_k=1)       # just the best match

    # ================================================================
    #  9. REINDEX — rebuild clusters
    # ================================================================
    # normally automatic (triggers on 10% data growth after 1000 vectors)
    # but you can force it manually

    # db.reindex()

    # ================================================================
    # 10. BACKUP & RESTORE — disaster recovery
    # ================================================================
    # backup pushes your local index to telegram as a file
    # restore pulls it back (use on a new machine)

    # db.backup()    # saves index to telegram
    # db.restore()   # downloads index from telegram (new machine setup)

    # ================================================================
    # 11. STATS — database info
    # ================================================================
    print("\n--- database stats ---")
    stats = db.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ================================================================
    # 12. LIST SOURCES — see what files have been added
    # ================================================================
    # only shows sources that are currently in cache
    # (query a few times first to populate cache)

    # sources = db.list_sources()
    # print("sources:", sources)

    # ================================================================
    # 13. DELETE — remove vectors by filter
    # ================================================================
    # deletes from both telegram and local index

    # db.delete(filter={"src": "outdated_doc.pdf"})   # remove all chunks from a file
    # db.delete(filter={"topic": "old_stuff"})         # remove by any metadata field

    # ================================================================
    # 14. USING WITH RAG — plug into any LLM
    # ================================================================
    # search returns text chunks + scores, perfect for stuffing into a prompt

    # def ask_llm(question):
    #     results = db.search(question, top_k=5)
    #     context = "\n\n".join([r["text"] for r in results])
    #     prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    #     # pass prompt to your LLM: ollama, gemini, openai, llama.cpp, etc
    #     return call_your_llm(prompt)
    # You can directly refer to chatbot.py for its implementation

    # ================================================================
    # 15. CONTEXT MANAGER — auto cleanup
    # ================================================================
    # can also use as a context manager for automatic close

    # with TgVectorDB(api_id=API_ID, api_hash=API_HASH, phone=PHONE,
    #                  db_name="quickstart-demo") as db:
    #     results = db.search("something")


    # ================================================================
    #  DONE
    # ================================================================
    db.close()
    print("\ndone!")


if __name__ == "__main__":
    main()