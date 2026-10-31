import os
import json
import pandas as pd

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE ONLY THESE TWO VALUES FOR EACH NEW BATCH
START_ROW = 5501
END_ROW = 5600

INPUT_FILE = "data/odia_literature.csv"
OUTPUT_FILE = "output/pilot_translated.csv"

# MODEL = "gemini-3.1-flash-lite"
MODEL = "gemini-3.5-flash-lite"



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
# TRANSLATION INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are a professional Odia to English translator.

Your task is to translate Odia sentences into accurate, natural English
for a machine translation training dataset.

STRICT RULES:

1. Preserve the exact meaning of the Odia sentence.
2. Do not add information that is not present in the Odia.
3. Do not remove information that is present in the Odia.
4. Do not explain the sentence.
5. Do not summarize the sentence.
6. Do not interpret beyond the original meaning.
7. Natural English is preferred over awkward word-for-word translation.
8. Preserve names, places, religious terms, cultural terms and numbers.
9. Preserve tense and grammatical meaning.
10. One Odia sentence must produce exactly one English sentence.
11. Return only the requested JSON.
"""


# ============================================================
# TRANSLATE BATCH
# ============================================================

def translate_batch(sentences):

    input_data = []

    for index, sentence in sentences:

        input_data.append({
            "id": int(index),
            "odia": sentence
        })

    prompt = f"""
Translate the following Odia sentences into English.

Return ONLY a JSON array in this format:

[
    {{
        "id": 101,
        "english": "English translation"
    }}
]

Every ID must appear exactly once.

Do not skip any ID.

Do not create any extra IDs.

Sentences:

{json.dumps(input_data, ensure_ascii=False)}
"""

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


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows in dataset: {len(df)}")

    # --------------------------------------------------------
    # VALIDATE ROW RANGE
    # --------------------------------------------------------

    if START_ROW < 1:
        raise ValueError("START_ROW must be 1 or greater.")

    if END_ROW < START_ROW:
        raise ValueError("END_ROW must be greater than or equal to START_ROW.")

    if END_ROW > len(df):
        raise ValueError(
            f"END_ROW {END_ROW} exceeds dataset size {len(df)}."
        )

    # --------------------------------------------------------
    # SELECT ROWS
    # --------------------------------------------------------
    #
    # User-facing row numbers:
    # 1 -> iloc[0]
    # 101 -> iloc[100]
    #
    # Therefore subtract 1 from START_ROW.
    # END_ROW is used directly because iloc end is exclusive.
    # --------------------------------------------------------

    pilot = df.iloc[START_ROW - 1:END_ROW].copy()

    print(
        f"Using rows {START_ROW} to {END_ROW} "
        f"for translation..."
    )

    print(f"Number of rows in this batch: {len(pilot)}")

    # --------------------------------------------------------
    # CREATE ORIGINAL ROW ID
    # --------------------------------------------------------

    pilot.insert(
        0,
        "id",
        range(START_ROW, END_ROW + 1)
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------
    # PREPARE SENTENCES
    # --------------------------------------------------------

    sentences = []

    for _, row in pilot.iterrows():

        odia_text = str(row["text"]).strip()

        sentences.append(
            (row["id"], odia_text)
        )

    # --------------------------------------------------------
    # TRANSLATE
    # --------------------------------------------------------

    print("Sending sentences for translation...")

    translations = translate_batch(sentences)

    # --------------------------------------------------------
    # CONVERT TRANSLATIONS TO DATAFRAME
    # --------------------------------------------------------

    translation_df = pd.DataFrame(translations)

    # --------------------------------------------------------
    # CHECK THAT ALL IDS WERE RETURNED
    # --------------------------------------------------------

    expected_ids = set(pilot["id"])
    received_ids = set(translation_df["id"])

    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    if missing_ids:
        raise ValueError(
            f"Gemini did not return these IDs: {sorted(missing_ids)}"
        )

    if extra_ids:
        raise ValueError(
            f"Gemini returned unexpected IDs: {sorted(extra_ids)}"
        )

    # --------------------------------------------------------
    # MERGE ODIA + ENGLISH
    # --------------------------------------------------------

    final_df = pilot[["id", "text"]].merge(
        translation_df,
        on="id",
        how="left"
    )

    # --------------------------------------------------------
    # RENAME COLUMN
    # --------------------------------------------------------

    final_df.rename(
        columns={
            "text": "odia"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # APPEND TO EXISTING OUTPUT
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        print("Existing output file found.")

        old_df = pd.read_csv(OUTPUT_FILE)

        # Prevent accidental duplicate IDs
        existing_ids = set(old_df["id"])

        duplicate_ids = existing_ids.intersection(
            set(final_df["id"])
        )

        if duplicate_ids:

            raise ValueError(
                f"These IDs already exist in output file: "
                f"{sorted(duplicate_ids)}\n"
                f"Change START_ROW and END_ROW."
            )

        # Append new translations
        final_df = pd.concat(
            [old_df, final_df],
            ignore_index=True
        )

    # --------------------------------------------------------
    # SORT BY ORIGINAL ID
    # --------------------------------------------------------

    final_df.sort_values(
        by="id",
        inplace=True
    )

    # Reset index
    final_df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n========================================")
    print("Translation completed successfully!")
    print("========================================")

    print(f"Translated this batch: {START_ROW} to {END_ROW}")
    print(f"Rows in this batch: {len(pilot)}")
    print(f"Total rows in output: {len(final_df)}")
    print(f"Output saved to: {OUTPUT_FILE}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()