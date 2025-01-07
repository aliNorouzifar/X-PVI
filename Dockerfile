
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy application code and requirements
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8050

# Command to run the application
CMD ["python", "app.py"]
