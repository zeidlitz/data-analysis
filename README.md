## data analysis

python3 service that consumes data from a redis stream and runs keyword extraction using KeyBERT and runs a sentiment analysis of the keywords using a huggingface pipeline
                                                            
```
┌─────┐    ┌─────────────┐    ┌─────┐
│redis│ <- │data-analysis│ -> │redis│
└─────┘    └─────────────┘    └─────┘
```

### configuration

reads the following configuration from `config.yaml` expected to be found at `/etc/data-analysis/config.yaml`

| name | description |
| ---- | ----------- |
| CONSUMER_STREAM | name of the redis stream to consume from |
| PRODUCER_STREAM | name of the redis stream to publish to |
| CONSUMER_GROUP  | name of the service consumer group in redis |
| CONSUMER_NAME   | service consumer name in redis |
| REDIS_HOST      | hostname for redis instance to connect to |
| REDIS_PORT      | port number for redis instance to connect  to |

### dependencies
```bash
pip install -r requierments.txt
python3 -m spacy download en_core_web_sm
```
