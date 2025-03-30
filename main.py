import json
import yaml
import redis
import logging
import spacy

from keybert import KeyBERT
from transformers import pipeline

CONFIG_PATH = "/etc/data-analysis/config.yaml"
VERSION_FILE = "VERSION"

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

config = load_config(CONFIG_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")
sentiment_pipeline = pipeline("sentiment-analysis")

CONSUMER_STREAM = config.get("consumer_stream", "data_extraction")
PRODUCER_STREAM = config.get("producer_stream ","data_analysis")
CONSUMER_GROUP = config.get("consumer_group ","data_analysis")
CONSUMER_NAME = config.get("consumer_name","analysis")
REDIS_HOST = config.get("redis", {}).get("host", "localhost")
REDIS_PORT = config.get("redis", {}).get("port", 6379)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


def create_redis_consumer_group():
    try:
        logging.info(f"Creating consumer group {CONSUMER_GROUP} for {CONSUMER_STREAM}")
        redis_client.xgroup_create(CONSUMER_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
    except Exception as e:
        logging.info(f"Exception {e}")
        pass


def categorize_text(text):
    doc = nlp(text)
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
        categories = categorize_text(entry["body"])
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

def get_version():
    with open("VERSION") as f:
        return f.read().strip()

def main():
    __version__ = get_version()
    logging.info(f"Running version {__version__}")
    create_redis_consumer_group()
    while True:
        data = consume_stream()
        output_data = analyze_data(data)
        publish_data(output_data)

if __name__ == "__main__":
    main()
