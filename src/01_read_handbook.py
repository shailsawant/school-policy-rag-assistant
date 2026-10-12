from pathlib import Path
from pypdf import PdfReader

# Locate the project folder, regardless of where we run the script.
project_folder = Path(__file__).resolve().parent.parent

pdf_path = project_folder / "data" / "school_handbook.pdf"

reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))

# Read the first page as a quick check.
# first_page_text = reader.pages[0].extract_text()

# print("\nFirst page text:\n")
# print(first_page_text)

# Printed page 39 corresponds to index 38.
page = reader.pages[38]
page_text = page.extract_text() or ""

print("\nAttendance policy page:\n")
print(page_text)