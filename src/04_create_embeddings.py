import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

with (data_folder / "chunks.json").open("r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save(data_folder / "embeddings.npy", embeddings)

print("\nEmbedding shape:", embeddings.shape)
print("First embedding, first 5 numbers:", embeddings[0][:5])
print("Saved embeddings.npy")