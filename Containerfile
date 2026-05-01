# build
FROM python:3.11-slim AS build
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --prefix=/install .

# run
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /install /usr/local
RUN python3 -m spacy download en_core_web_sm
COPY . .
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
CMD ["python", "main.py"]
