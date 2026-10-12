import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

project_folder = Path(__file__).resolve().parent.parent
load_dotenv(project_folder / ".env")

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

if not api_key or not model_name:
    raise ValueError("Set GEMINI_API_KEY and GEMINI_MODEL in .env")

# Handle retries ourselves instead of stacking them on SDK retries.
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

max_attempts = 3

try:
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Reply with exactly: Gemini connection successful"
            )

            if response.text:
                print(response.text)
            else:
                print("Gemini returned no text. Please try again.")

            break

        except errors.APIError as error:
            retryable = error.code in {500, 502, 503, 504}

            if retryable and attempt < max_attempts:
                wait_seconds = 5 * attempt
                print(
                    f"Gemini temporarily unavailable ({error.code}). "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

            if retryable:
                print("Gemini is unavailable after 3 attempts. Try again later.")
            elif error.code == 429:
                print("Gemini quota or rate limit reached. Check AI Studio usage.")
            elif error.code in {401, 403}:
                print("Gemini denied access. Check your API key and permissions.")
            elif error.code == 404:
                print("Model unavailable. Check GEMINI_MODEL in .env.")
            else:
                print(f"Gemini request failed with HTTP {error.code}.")

            break

finally:
    client.close()