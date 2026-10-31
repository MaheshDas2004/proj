import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE ONLY THESE TWO VALUES FOR EACH NEW BATCH
START_ROW = 11
END_ROW = 20

INPUT_FILE = "output/pilot_translated.csv"
OUTPUT_FILE = "output/pilot_checked.csv"

MODEL = "gemini-3.6-flash"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# QUALITY CHECK INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are a professional Odia-to-English translation quality evaluator.

Your task is to evaluate whether the English translation faithfully
preserves the meaning of the original Odia sentence.

Do NOT rewrite the translation.

Check the following:

1. Is the meaning preserved?
2. Is any information missing?
3. Has any information been added?
4. Are names and places preserved correctly?
5. Are numbers and dates preserved correctly?
6. Are religious and cultural terms handled correctly?
7. Is the tense and grammatical meaning preserved?
8. Is the English natural and understandable?
9. A natural semantic translation is acceptable.
10. Do not require word-for-word translation.

Use ONLY these labels:

PASS
The English translation correctly preserves the meaning.

REVIEW
There may be a translation, meaning, terminology, grammar,
context or other semantic problem.

REVIEW_SOURCE
The original Odia sentence itself appears incomplete or corrupted,
for example a missing number, year, word or other information.
Do NOT guess missing information.

Important:

- Judge the English against the original Odia.
- Do not mark a translation wrong merely because it is not word-for-word.
- Natural semantic translation is preferred.
- If the Odia source itself is incomplete, use REVIEW_SOURCE.
- If the source is complete but the English may have a meaning,
  terminology, grammar or context problem, use REVIEW.
- Return only valid JSON.
"""


# ============================================================
# CHECK BATCH
# ============================================================

def check_batch(df):

    input_data = []

    for _, row in df.iterrows():

        input_data.append({
            "id": int(row["id"]),
            "odia": str(row["odia"]),
            "english": str(row["english"])
        })

    prompt = f"""
Evaluate the following Odia-English translation pairs.

Return ONLY a JSON array in exactly this format:

[
    {{
        "id": 101,
        "status": "PASS",
        "reason": "The meaning is accurately preserved."
    }}
]

Allowed status values:

PASS
REVIEW
REVIEW_SOURCE

Every ID must appear exactly once.

Do not skip any ID.

Do not create any extra IDs.

Translation pairs:

{json.dumps(input_data, ensure_ascii=False)}
"""

    max_retries = 5

    for attempt in range(max_retries):

        try:

            print(
                f"Checking translations... "
                f"Attempt {attempt + 1}/{max_retries}"
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)

            return result

        except Exception as e:

            print(f"Request failed: {e}")

            if attempt == max_retries - 1:
                raise

            wait_time = 5 * (2 ** attempt)

            print(f"Retrying in {wait_time} seconds...")

            time.sleep(wait_time)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("Loading translated dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total translated rows: {len(df)}")


    # ========================================================
    # VALIDATE RANGE
    # ========================================================

    if START_ROW < 1:
        raise ValueError(
            "START_ROW must be 1 or greater."
        )

    if END_ROW < START_ROW:
        raise ValueError(
            "END_ROW must be greater than or equal to START_ROW."
        )


    # ========================================================
    # SELECT BY ORIGINAL ID
    # ========================================================

    selected_df = df[
        (df["id"] >= START_ROW) &
        (df["id"] <= END_ROW)
    ].copy()


    if len(selected_df) == 0:

        raise ValueError(
            f"No rows found with IDs {START_ROW} to {END_ROW}."
        )


    print(
        f"Checking IDs {START_ROW} to {END_ROW}..."
    )

    print(
        f"Number of rows in this batch: {len(selected_df)}"
    )


    # ========================================================
    # CHECK FOR MISSING IDS
    # ========================================================

    expected_ids = set(
        range(START_ROW, END_ROW + 1)
    )

    actual_ids = set(
        selected_df["id"]
    )

    missing_ids = expected_ids - actual_ids


    if missing_ids:

        raise ValueError(
            f"These IDs are missing from the translated dataset: "
            f"{sorted(missing_ids)}"
        )


    # ========================================================
    # RUN QUALITY CHECK
    # ========================================================

    results = check_batch(selected_df)

    check_df = pd.DataFrame(results)


    # ========================================================
    # VALIDATE GEMINI RESPONSE
    # ========================================================

    received_ids = set(
        check_df["id"]
    )

    missing_check_ids = expected_ids - received_ids

    extra_check_ids = received_ids - expected_ids


    if missing_check_ids:

        raise ValueError(
            f"Gemini did not return these IDs: "
            f"{sorted(missing_check_ids)}"
        )


    if extra_check_ids:

        raise ValueError(
            f"Gemini returned unexpected IDs: "
            f"{sorted(extra_check_ids)}"
        )


    # ========================================================
    # MERGE QUALITY RESULTS
    # ========================================================

    final_batch = selected_df.merge(
        check_df,
        on="id",
        how="left"
    )


    # ========================================================
    # APPEND TO EXISTING CHECKED OUTPUT
    # ========================================================

    os.makedirs("output", exist_ok=True)


    if os.path.exists(OUTPUT_FILE):

        print("Existing checked output found.")

        old_df = pd.read_csv(
            OUTPUT_FILE
        )


        # Prevent duplicate IDs

        existing_ids = set(
            old_df["id"]
        )

        duplicate_ids = existing_ids.intersection(
            set(final_batch["id"])
        )


        if duplicate_ids:

            raise ValueError(
                f"These IDs already exist in checked output: "
                f"{sorted(duplicate_ids)}\n"
                f"Change START_ROW and END_ROW."
            )


        # Append new checked rows

        final_df = pd.concat(
            [old_df, final_batch],
            ignore_index=True
        )

    else:

        final_df = final_batch


    # ========================================================
    # SORT BY ORIGINAL ID
    # ========================================================

    final_df.sort_values(
        by="id",
        inplace=True
    )

    final_df.reset_index(
        drop=True,
        inplace=True
    )


    # ========================================================
    # SAVE
    # ========================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("Semantic checking completed!")
    print("========================================")

    print(
        f"Checked this batch: {START_ROW} to {END_ROW}"
    )

    print(
        f"Rows checked: {len(final_batch)}"
    )

    print(
        f"Total checked rows: {len(final_df)}"
    )

    print(
        f"Output saved to: {OUTPUT_FILE}"
    )

    print("\nQuality summary:")

    print(
        final_batch["status"].value_counts()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()