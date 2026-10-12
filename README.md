# School Policy RAG Assistant

A school handbook question-answering demo built with Python, Streamlit,
ChromaDB, Sentence Transformers and the Gemini API.

It retrieves relevant handbook passages and generates short answers with
page citations. When the retrieved excerpts do not contain an answer, it
is instructed to say so.

**Live demo:** https://school-policy-rag-assistant.streamlit.app/

## Problem

Parents need to find school rules about attendance, early dismissal,
uniforms, devices and transport without searching an entire handbook.

This project demonstrates a Retrieval-Augmented Generation (RAG)
workflow for that task.

## How it works

1. Download the handbook from the school's website.
2. Extract selected policy pages.
3. Split the text into 180-word chunks with 30-word overlap.
4. Generate local embeddings using `all-MiniLM-L6-v2`.
5. Store passages, embeddings and page metadata in ChromaDB.
6. Retrieve the three closest passages using cosine distance.
7. Send those passages and the question to Gemini.
8. Display the answer, page citations and retrieved evidence.

The app caches its embedding model and in-memory ChromaDB collection.
It rebuilds the collection when the app process restarts.

## Technology

| Component | Technology |
|---|---|
| Interface | Streamlit |
| PDF extraction | pypdf |
| Embeddings | Sentence Transformers: all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Answer generation | Gemini API through google-genai |
| Local configuration | python-dotenv |
| Hosting | Streamlit Community Cloud |

## Project structure

```text
school-policy-rag-assistant/
├── dashboard/
│   └── app.py
├── src/
│   ├── 01_read_handbook.py
│   ├── 02_extract_policies.py
│   ├── 03_create_chunks.py
│   ├── 04_create_embeddings.py
│   ├── 05_build_vector_db.py
│   ├── 06_test_retrieval.py
│   ├── 07_test_gemini.py
│   └── 08_rag_answer.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `src` scripts demonstrate each stage separately.
The Streamlit app runs the complete workflow without requiring those
scripts to be run first.

Local `.env`, downloaded PDFs, generated JSON files, embedding files and
the on-disk database are excluded from Git.

## Run locally

Python 3.13 was used for development.

```powershell
git clone https://github.com/shailsawant/school-policy-rag-assistant.git
cd school-policy-rag-assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Create `.env` in the project root:

```dotenv
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.8-flash
```

Create an API key at https://aistudio.google.com/apikey.
Model access and quotas depend on your Google project.

Start the app:

```powershell
python -m streamlit run dashboard/app.py
```

The first question requires downloading the handbook and embedding model.
An internet connection is required.

To run the individual learning scripts, first download the source PDF
and save it as `data/school_handbook.pdf`. Run scripts 01–08 in order.
Scripts 01–06 do not require a Gemini key.

## Deploy to Streamlit Community Cloud

Use these deployment settings:

- Repository: `shailsawant/school-policy-rag-assistant`
- Branch: `main`
- Main file: `dashboard/app.py`
- Python version: `3.13`

Add these values in Streamlit's Secrets settings:

```toml
GEMINI_API_KEY = "your_api_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Never commit API keys or secrets to GitHub.

## Manual checks

The following questions were checked locally and on the deployed app:

| Question | Observed result |
|---|---|
| What should I do if my child needs to leave school early? | Answer supported by page 39, with citations |
| Does the school refund fees if my child withdraws halfway through the year? | Declined to answer from the available excerpts |

These are limited manual checks, not a comprehensive accuracy evaluation.

Temporary Gemini server errors are retried up to three times.
API and network failures are shown as readable messages in the interface.

## Source and attribution

Source: Sreenidhi International School, Student–Parent Handbook,
updated January 2025.

https://sis.edu.in/wp-content/uploads/2025/02/STUDENT-PARENT-HANDBOOK.pdf

Selected printed pages: 34, 35 and 38–46.

The PDF is fetched from the original website at runtime and is not
included in this repository. Handbook content belongs to its respective
rights holders.

This is an independent educational project, unaffiliated with and not
endorsed by the school.

## Limitations

- Only selected sections of the January 2025 handbook are searched.
- The source may differ from current school policies.
- Retrieval can miss relevant passages, especially across page boundaries.
- Citations and refusal instructions do not guarantee factual accuracy.
- Source URL, PDF layout or model availability changes may break the app.
- Gemini capacity and quota limits can interrupt answers.

Verify important policy decisions directly with the school.