import json
from pathlib import Path
from pypdf import PdfReader

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

reader = PdfReader(data_folder / "school_handbook.pdf")

source_url = (
    "https://sis.edu.in/wp-content/uploads/2025/02/"
    "STUDENT-PARENT-HANDBOOK.pdf"
)

# Printed page numbers. Include neighbouring pages for continued rules.
selected_pages = [34, 35, 38, 39, 40, 41, 42, 43, 44, 45, 46]

policies = []

for page_number in selected_pages:
    raw_text = reader.pages[page_number - 1].extract_text() or ""

    # Remove empty lines, standalone backticks and page-number footers.
    lines = []
    for line in raw_text.splitlines():
        clean_line = " ".join(line.split())

        if clean_line and clean_line not in {"`", str(page_number)}:
            lines.append(clean_line)

    text = "\n".join(lines)

    if not text:
        raise ValueError(f"No text extracted from page {page_number}")

    policies.append({
        "page": page_number,
        "text": text,
        "source_url": source_url
    })

    print(f"Page {page_number}: {len(text)} characters")

output_path = data_folder / "policies.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(policies, file, ensure_ascii=False, indent=2)

print(f"\nSaved {len(policies)} pages to {output_path.name}")