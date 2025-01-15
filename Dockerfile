

FROM python:3.10

# Set environment variables for the JDK
ENV JAVA_HOME=/usr/lib/jdk-21
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install necessary tools
RUN apt-get update && apt-get install -y wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install JDK 21
RUN wget -q https://download.java.net/java/GA/jdk21/ri/openjdk-21+35_linux-x64_bin.tar.gz -O /tmp/jdk-21.tar.gz && \
    mkdir -p /usr/lib/jdk-21 && \
    tar -xzf /tmp/jdk-21.tar.gz -C /usr/lib/jdk-21 --strip-components=1 && \
    rm -f /tmp/jdk-21.tar.gz

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["python", "app.py"]

LABEL org.opencontainers.image.source="https://github.com/alinorouzifar/x-pvi"

ENV PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH"