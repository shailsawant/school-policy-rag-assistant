import json
from pathlib import Path

import chromadb
import numpy as np

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

with (data_folder / "chunks.json").open("r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(data_folder / "embeddings.npy")

if len(chunks) != len(embeddings):
    raise ValueError("Chunk count and embedding count do not match")

client = chromadb.PersistentClient(
    path=str(project_folder / "chroma_db")
)

collection = client.get_or_create_collection(
    name="school_policies",
    embedding_function=None,
    configuration={"hnsw": {"space": "cosine"}}
)

collection.upsert(
    ids=[chunk["id"] for chunk in chunks],
    documents=[chunk["text"] for chunk in chunks],
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "page": chunk["page"],
            "source_url": chunk["source_url"]
        }
        for chunk in chunks
    ]
)

print("Stored chunks:", collection.count())
print("Database folder:", project_folder / "chroma_db")