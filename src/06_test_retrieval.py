from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

project_folder = Path(__file__).resolve().parent.parent

client = chromadb.PersistentClient(
    path=str(project_folder / "chroma_db")
)

collection = client.get_collection(
    name="school_policies",
    embedding_function=None
)

# Use the same model that created the document embeddings.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

question = input("Ask a school policy question: ").strip()

if not question:
    raise ValueError("Please enter a question")

question_embedding = model.encode(
    [question],
    normalize_embeddings=True
)

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for index in range(len(results["ids"][0])):
    text = results["documents"][0][index]
    metadata = results["metadatas"][0][index]
    distance = results["distances"][0][index]

    print(f"\nResult {index + 1}")
    print("Page:", metadata["page"])
    print(f"Cosine distance: {distance:.4f}")
    print("Text:", text)
    print("Source:", metadata["source_url"])