"""
TgVectorDB - Free, unlimited vector database backed by Telegram.

basically you store embeddings as telegram messages and search over them.
no servers, no bills, just your telegram account.

usage:
    from tgvectordb import TgVectorDB

    db = TgVectorDB(api_id=..., api_hash=..., phone="+91xxx")
    db.add("some text about dogs")
    db.add_source("research_paper.pdf")
    db.add_source("notes.docx")
    db.add_directory("./my_docs/")
    results = db.search("tell me about puppies", top_k=5)
"""

from tgvectordb.client import TgVectorDB
from tgvectordb.ingestors.registry import ingest, is_supported, list_supported_formats

__version__ = "0.1.0"
__all__ = ["TgVectorDB", "ingest", "is_supported", "list_supported_formats"]
