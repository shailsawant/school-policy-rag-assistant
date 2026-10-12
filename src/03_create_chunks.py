import json
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

with (data_folder / "policies.json").open("r", encoding="utf-8") as file:
    policies = json.load(file)

chunk_size = 180
overlap = 30
step = chunk_size - overlap

chunks = []

for policy in policies:
    words = policy["text"].split()
    start = 0
    chunk_number = 1

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "id": f"page_{policy['page']}_chunk_{chunk_number}",
            "text": chunk_text,
            "page": policy["page"],
            "source_url": policy["source_url"]
        })

        # Stop once this chunk includes the page's final word.
        if end >= len(words):
            break

        start += step
        chunk_number += 1

output_path = data_folder / "chunks.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Total chunks:", len(chunks))
print("\nFirst chunk ID:", chunks[0]["id"])
print("Page:", chunks[0]["page"])
print("Word count:", len(chunks[0]["text"].split()))
print("\nText:\n", chunks[0]["text"])