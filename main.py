import json
import redis
import logging
import spacy

from keybert import KeyBERT
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")

STREAM_NAME = "data_extraction"
OUTPUT_DATA = "output.json"
CONSUMER_GROUP = "data_analysis"
CONSUMER_NAME = "analysis"
redis_consumer = redis.Redis(host='localhost', port=6379, decode_responses=True)

def extract_categories_with_context(text, context):
    doc = nlp(text + " " + context)
    entities = {ent.text.upper() for ent in doc.ents}
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    keyphrases = {kw[0].upper() for kw in keywords}
    categories = list(entities | keyphrases)
    split_list = [word for item in categories[:5] for word in item.split()]
    return list(set(split_list))

def consume_stream():
    while True:
        try:
            messages = redis_consumer.xreadgroup(CONSUMER_GROUP, CONSUMER_NAME, {STREAM_NAME: '>'}, count=1, block=5000)
            logging.info(f"Consuming messages from {STREAM_NAME}")
            if messages:
                logging.info(f"Message received!")
                for stream, message_list in messages:
                    for message_id, message in message_list:
                        redis_consumer.xack(STREAM_NAME, CONSUMER_GROUP, message_id)
                        return json.loads(message['data'])
        except Exception as e:
            print(f"Error: {e}")

def analyze_data(data):
    output_data = []
    for entry in data:
        categories = extract_categories_with_context(entry["body"], entry["posted_in"])
        sentiment_result = sentiment_pipeline(entry["body"])[0]
        output_entry = {
                "source": entry["source"],
                "subsource": entry["subsource"],
                "unix_timestamp": entry["unix_timestamp"],
                "posted_in": entry["posted_in"],
                "category": categories,
                "sentiment": sentiment_result["label"]
        }
        output_data.append(output_entry)

    logging.info(f"Writing to {OUTPUT_DATA}")
    with open(OUTPUT_DATA, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    logging.info("Analysis complete")


def main():
    while True:
        data = consume_stream()
        analyze_data(data)

if __name__ == "__main__":
    main()
