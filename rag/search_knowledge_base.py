
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import argparse

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = PROJECT_ROOT / "rag" / "chroma_db"

COLLECTION_NAME = "industrial_quality_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the Sentence Transformers embedding model once
    per Python process.

    Subsequent searches reuse the same model.
    """

    print("\nLoading embedding model into cache...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


# ============================================================
# CACHED CHROMADB CONNECTION
# ============================================================

@lru_cache(maxsize=1)
def get_knowledge_collection():
    """
    Reuse the ChromaDB client and collection.

    If the knowledge base is rebuilt while this process
    is running, restart the application to refresh it.
    """

    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge database not found: {DATABASE_DIR}"
        )

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception as error:
        raise RuntimeError(
            "Knowledge collection not found. "
            "Run: python rag/build_knowledge_base.py"
        ) from error

    return client, collection


# ============================================================
# KNOWLEDGE BASE SEARCH
# ============================================================

def search_knowledge_base(query, top_k=3):
    """
    Search the local quality-control knowledge base.

    Returns a list of dictionaries containing:
        source
        distance
        content

    This preserves compatibility with rag_assistant.py.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query must be a non-empty string."
        )

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            "top_k must be a positive integer."
        )

    # --------------------------------------------------------
    # GET CACHED RESOURCES
    # --------------------------------------------------------

    model = get_embedding_model()

    _, collection = get_knowledge_collection()

    document_count = collection.count()

    if document_count == 0:
        return []

    # --------------------------------------------------------
    # GENERATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(
        query.strip(),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # --------------------------------------------------------
    # RETRIEVE DOCUMENT SECTIONS
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, document_count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        matches.append(
            {
                "source": metadata["source"],
                "distance": float(distance),
                "content": document,
            }
        )

    return matches


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Search the Industrial VisionGuard knowledge base"
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to search for",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of document sections to retrieve",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("INDUSTRIAL VISIONGUARD - RAG SEARCH")
    print("=" * 60)

    print(f"\nQuestion: {args.query}")

    start = perf_counter()

    matches = search_knowledge_base(
        query=args.query,
        top_k=args.top_k,
    )

    elapsed = perf_counter() - start

    for index, match in enumerate(
        matches,
        start=1,
    ):

        print(f"\n{'-' * 60}")
        print(f"RESULT {index}")
        print(f"Source: {match['source']}")
        print(
            f"Cosine distance: "
            f"{match['distance']:.4f}"
        )
        print(f"{'-' * 60}")
        print(match["content"])

    print("\n" + "=" * 60)
    print(f"Retrieved sections: {len(matches)}")
    print(f"Search time: {elapsed:.4f} seconds")
    print("Search complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
