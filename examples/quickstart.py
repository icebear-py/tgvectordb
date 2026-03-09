from tgvectordb import TgVectorDB
import subprocess
import sys


# replace these with your actual credentials
API_ID = 35950289
API_HASH = "fee73dd00efce76ec3f20ec04984fc90"
PHONE = "+918755868585"


def main():
    # create / connect to database
    db = TgVectorDB(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        db_name="quickstart-demo",
    )

    # add some sample data
    print("\n--- adding sample data ---")
    db.add(
        "The mitochondria is the powerhouse of the cell",
        metadata={"src": "biology", "topic": "cells"},
    )
    db.add(
        "Python is a high-level programming language known for its simplicity",
        metadata={"src": "programming", "topic": "python"},
    )
    db.add(
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
        metadata={"src": "biology", "topic": "plants"},
    )
    db.add(
        "Machine learning models learn patterns from training data",
        metadata={"src": "programming", "topic": "ml"},
    )
    db.add(
        "DNA carries the genetic instructions for all living organisms",
        metadata={"src": "biology", "topic": "genetics"},
    )

    # search
    print("\n--- searching: 'how do plants make energy?' ---")
    results = db.search("how do plants make energy?", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")
        print(f"          metadata: {r['metadata']}")

    print("\n--- searching: 'what is python?' ---")
    results = db.search("what is python?", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")

    # search with filter
    print("\n--- searching 'learning' but only in programming ---")
    results = db.search("learning", top_k=3, filter={"src": "programming"})
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")

    # stats
    print("\n--- database stats ---")
    stats = db.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    db.close()
    print("\ndone!")


if __name__ == "__main__":
    main()
