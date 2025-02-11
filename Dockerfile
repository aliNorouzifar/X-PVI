# Use Python base image
FROM python:3.10

# Set environment variables for the JDK
ENV JAVA_HOME=/usr/lib/jvm/jdk-21
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install necessary tools and Redis
RUN apt-get update && apt-get install -y wget redis && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install JDK 21 using the .deb package
RUN wget -q https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb -O /tmp/jdk-21.deb && \
    apt-get install -y /tmp/jdk-21.deb && \
    rm -f /tmp/jdk-21.deb


# Set working directory
WORKDIR /

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose application and Redis ports
EXPOSE 8002 6379

# Start Redis and the Python application
CMD ["bash", "-c", "service redis-server start && python app.py"]

# Metadata for the Docker image
LABEL org.opencontainers.image.source="https://github.com/alinorouzifar/x-pvi"

# Update PATH for JDK
ENV PATH="/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH"