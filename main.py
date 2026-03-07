import os
import yaml
import redis
import logging
import sys
import spacy
from importlib.metadata import version, PackageNotFoundError
from data_models.v1.data_models_pb2 import RedditData, AnalysisResult

from keybert import KeyBERT
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParseException(Exception):
    pass


def get_version():
    __version__ = "unknown"
    try:
        __version__ = version("data-extraction")
    except PackageNotFoundError:
        logging.warning(
            "could not read pacakge version, ensure project is installed properly"
        )
    return __version__


def parse_args():
    args = sys.argv
    nrArgs = len(args)
    if nrArgs == 1:
        return "config.yaml"
    if nrArgs > 2:
        raise ParseException(
            f"too many input arguments, recieved {nrArgs} need exactlly one. Recieved: {args[1:]}"
        )
    return args[1]


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def create_redis_client(host, port, decode_responses) -> redis.Redis:
    return redis.Redis(host=host, port=port, decode_responses=decode_responses)


def create_redis_consumer_group(
    redis_client, consumer_stream, consumer_group, _id, mkstream
):
    try:
        logging.info(f"Creating consumer group {consumer_group} for {consumer_stream}")
        redis_client.xgroup_create(
            consumer_stream, consumer_group, id=_id, mkstream=mkstream
        )
    except Exception as e:
        logging.info(f"Exception {e}")
        pass


def categorize_text(nlp, kw_model, text):
    doc = nlp(text)
    entities = {ent.text.upper() for ent in doc.ents}
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words="english"
    )
    keyphrases = {kw[0].upper() for kw in keywords}
    categories = list(entities | keyphrases)
    split_list = [word for item in categories[:5] for word in item.split()]
    return list(set(split_list))


def consume_stream(redis_client, consumer_group, consumer_name, consumer_stream):
    while True:
        try:
            messages = redis_client.xreadgroup(
                consumer_group,
                consumer_name,
                {consumer_stream: ">"},
                count=1,
                block=5000,
            )
            if messages:
                logging.info(f"message received from {consumer_stream}")
                for _, message_list in messages:
                    for message_id, message in message_list:
                        raw_data = message.get(b"data") or message.get("data")
                        data = RedditData()
                        data.ParseFromString(raw_data)
                        redis_client.xack(consumer_stream, consumer_group, message_id)
                        return data
        except Exception as e:
            logging.error(f"error consuming from stream: {e}")
            return None


def analyze_data(data, nlp, kw_model, sentiment_pipeline):
    analysis_result = AnalysisResult()
    categories = categorize_text(nlp, kw_model, data.body)
    sentiment_result = sentiment_pipeline(data.body)[0]
    analysis_result.raw_data.CopyFrom(data)
    analysis_result.categories.extend(categories)
    analysis_result.sentiment = sentiment_result["label"]
    return analysis_result

def publish_data(redis_client, producer_stream, data):
    serialized_data = data.SerializeToString()
    logging.info(f"publishing {len(serialized_data)} bytes to {producer_stream}")
    redis_client.xadd(producer_stream, {"data": serialized_data})

def main():
    __version__ = get_version()
    logging.info(f"Running version {__version__}")
    try:
        config_path = parse_args()
    except ParseException as e:
        print(e)
        os._exit(1)

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"could not open config: {e} ")
        os._exit(1)

    consumer_stream = config.get("consumer_stream", "data_extraction")
    producer_stream = config.get("producer_stream ", "data_analysis")
    consumer_group = config.get("consumer_group ", "data_analysis")
    consumer_name = config.get("consumer_name", "analysis")
    redis_host = config.get("redis", {}).get("host", "localhost")
    redis_port = config.get("redis", {}).get("port", 6379)
    redis_client = create_redis_client(redis_host, redis_port, False)
    create_redis_consumer_group(redis_client, consumer_stream, consumer_group, 0, True)

    kw_model = KeyBERT()
    nlp = spacy.load("en_core_web_sm")
    sentiment_pipeline = pipeline("sentiment-analysis")

    while True:
        try:
            data = consume_stream(
                redis_client, consumer_group, consumer_name, consumer_stream
            )

            if data == None:
                logging.info("no data consumed")
                continue

        except Exception as e:
            logging.error(e)
            continue
        output_data = analyze_data(data, nlp, kw_model, sentiment_pipeline)
        publish_data(redis_client, producer_stream, output_data)
        logging.info("publishing data", output_data)


if __name__ == "__main__":
    main()
