import logging
import json
from transformers import pipeline
from tqdm import tqdm
import spacy
from transformers import pipeline
from keybert import KeyBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")

DATA_SOURCE = "data.json"
OUTPUT_DATA = "output.json"


def extract_categories(text, context):
    doc = nlp(text + " " + context)
    entities = {ent.text.upper() for ent in doc.ents}
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    keyphrases = {kw[0].upper() for kw in keywords}
    categories = list(entities | keyphrases)
    return categories[:5]


if __name__ == "__main__":
    with open(DATA_SOURCE, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    output_data = []
    for entry in tqdm(input_data, desc="Running analysis ..."):
        categories = extract_categories(entry["body"], entry["posted_in"])
        sentiment_result = sentiment_pipeline(entry["body"])[0]
        output_entry = {
                "unix_timestamp": entry["unix_timestamp"],
                "utc_timestamp": entry["utc_timestamp"],
                "posted_in": entry["posted_in"],
                "category": categories,
                "label": sentiment_result["label"]
        }
        output_data.append(output_entry)

    logging.info(f"Writing to {OUTPUT_DATA}")
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    logging.info("Analysis complete")
