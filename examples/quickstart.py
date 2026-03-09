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


    # search
    print("\n--- searching: 'What was the budget of my project EvalAI?' ---")
    results = db.search("What was the budget of my project EvalAI?", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")
        print(f"          metadata: {r['metadata']}")

    print("\n--- searching: 'who is ansh?' ---")
    results = db.search("who is ansh?", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:70]}...")

    print("\n--- searching: 'What was the budget of my project TestkaaroAI?' ---")
    results = db.search("What was the budget of my project TestkaaroAI?", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text']}...")
        print(f"          metadata: {r['metadata']}")


    # stats
    print("\n--- database stats ---")
    stats = db.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    db.close()
    print("\ndone!")


if __name__ == "__main__":
    main()
