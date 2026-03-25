from tgvectordb import TgVectorDB
import google.generativeai as genai


# ── your credentials ──
API_ID = 000000
API_HASH = ""
PHONE = "+91xxxxxxxxxx"
GEMINI_API_KEY = ""  # get from https://aistudio.google.com/apikey


# ── setup ──
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

db = TgVectorDB(
    api_id=API_ID,
    api_hash=API_HASH,
    phone=PHONE,
    db_name="quickstart-demo",
)


def ask(question, top_k=5, show_context=False):
    """
    search the vector db for relevant chunks,
    stuff them into a prompt, and ask gemini.
    """
    # retrieve
    results = db.search(question, top_k=top_k)

    if not results:
        return "no relevant context found in the database."

    # build context from search results
    context_parts = []
    for i, r in enumerate(results):
        src = r["metadata"].get("src", "unknown")
        score = r["score"]
        context_parts.append(f"[Source: {src} | Relevance: {score:.2f}]\n{r['text']}")

    context = "\n\n---\n\n".join(context_parts)

    if show_context:
        print("\n📎 retrieved context:")
        print("-" * 50)
        for i, r in enumerate(results):
            print(f"  [{r['score']:.3f}] {r['text'][:100]}...")
            print(f"          src: {r['metadata'].get('src', '?')}")
        print("-" * 50)

    # prompt
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context provided below. 
If the context doesn't contain enough information to answer, say so honestly - don't make stuff up.
Cite which source document the information came from when possible.

Context:
{context}

Question: {question}

Answer:"""

    # ask gemini
    response = model.generate_content(prompt)
    return response.text


def chat():
    """interactive chat loop."""
    print("\n" + "=" * 60)
    print("  TgVectorDB RAG Chatbot (powered by Gemini)")
    print("  ask questions about your documents")
    print("  type 'quit' to exit, 'context on/off' to toggle source display")
    print("=" * 60)

    show_ctx = False

    while True:
        print()
        question = input("you: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            break

        if question.lower() == "context on":
            show_ctx = True
            print("  context display: ON")
            continue

        if question.lower() == "context off":
            show_ctx = False
            print("  context display: OFF")
            continue

        if question.lower() == "stats":
            stats = db.stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue

        try:
            answer = ask(question, show_context=show_ctx)
            print(f"\n🤖 {answer}")
        except Exception as e:
            print(f"\n❌ error: {e}")

    db.close()
    print("bye!")


if __name__ == "__main__":
    chat()