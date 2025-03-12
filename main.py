import os
import json
import redis
import logging
import spacy

from dotenv import load_dotenv
from keybert import KeyBERT
from transformers import pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")

CONSUMER_STREAM = os.getenv("CONSUMER_STREAM", "data_extraction")
PRODUCER_STREAM = os.getenv("PRODUCER_STREAM ","data_analysis")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP ","data_analysis")
CONSUMER_NAME = os.getenv("CONSUMER_NAME","analysis")
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

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
            messages = redis_client.xreadgroup(CONSUMER_GROUP, CONSUMER_NAME, {CONSUMER_STREAM: '>'}, count=1, block=5000)
            logging.info(f"Consuming messages from {CONSUMER_STREAM}")
            if messages:
                logging.info(f"Message received!")
                for stream, message_list in messages:
                    for message_id, message in message_list:
                        redis_client.xack(CONSUMER_STREAM, CONSUMER_GROUP, message_id)
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
    return json.dumps(output_data)

def publish_data(data):
    logging.info(f"Publishing {len(data)} to {REDIS_HOST}:{REDIS_PORT}:{PRODUCER_STREAM}")
    redis_client.xadd(PRODUCER_STREAM, {"data": data})


def main():
    while True:
        data = consume_stream()
        output_data = analyze_data(data)
        publish_data(output_data)

if __name__ == "__main__":
    main()
