

FROM python:3.10

# Set environment variables for the JDK
ENV JAVA_HOME=/usr/lib/jvm/jdk-21
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install necessary tools
RUN apt-get update && apt-get install -y wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install JDK 21 using the .deb package
RUN wget -q https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb -O /tmp/jdk-21.deb && \
    apt-get install -y /tmp/jdk-21.deb && \
    rm -f /tmp/jdk-21.deb


WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8002

CMD ["python", "app.py"]

LABEL org.opencontainers.image.source="https://github.com/alinorouzifar/x-pvi"

ENV PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH"