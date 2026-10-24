
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "rag" / "documents"

DATABASE_DIR = PROJECT_ROOT / "rag" / "chroma_db"

COLLECTION_NAME = "industrial_quality_knowledge"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# DOCUMENT LOADING
# --------------------------------------------------

def load_documents():
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.md")):

        content = file_path.read_text(encoding="utf-8")

        sections = content.split("\n## ")

        for index, section in enumerate(sections):

            section = section.strip()

            if not section:
                continue

            # Skip document title-only sections
            if index == 0 and section.startswith("# "):
                continue

            if index > 0:
                section = "## " + section

            # Skip extremely short sections
            if len(section.split()) < 10:
                continue

            documents.append(
                {
                    "id": f"{file_path.stem}_{index}",
                    "text": section,
                    "source": file_path.name,
                }
            )

    return documents


# --------------------------------------------------
# KNOWLEDGE BASE BUILDER
# --------------------------------------------------

def main():

    print("=" * 60)
    print("INDUSTRIAL VISIONGUARD - KNOWLEDGE BASE BUILDER")
    print("=" * 60)

    print("\nLoading documents...")

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            f"No usable Markdown sections found in {DOCUMENTS_DIR}"
        )

    print(f"Loaded {len(documents)} document sections.")

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        document["text"]
        for document in documents
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )

    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:

        print("\nRemoving previous collection...")

        client.delete_collection(COLLECTION_NAME)

    print("\nCreating improved knowledge base...")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[
            document["id"]
            for document in documents
        ],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": document["source"],
                "embedding_model": EMBEDDING_MODEL,
            }
            for document in documents
        ],
    )

    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE REBUILT SUCCESSFULLY")
    print("=" * 60)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed sections: {collection.count()}")
    print(f"Database: {DATABASE_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
