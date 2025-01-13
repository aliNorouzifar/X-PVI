

FROM python:3.10

RUN apt-get update && apt-get install -y openjdk-11-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["python", "app.py"]

LABEL org.opencontainers.image.source="https://github.com/alinorouzifar/x-pvi"

ENV PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH"