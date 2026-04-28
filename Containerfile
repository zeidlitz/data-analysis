# build
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update
RUN apt-get install git -y
COPY . .
RUN pip install --no-cache-dir .
RUN python3 -m spacy download en_core_web_sm

# run
CMD ["python", "main.py"]
