# Step 1: Use an official Python runtime as a base image
FROM python:3.10

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy project files into the container
COPY . .

# Step 4: Install dependencies
# Make sure your project has a requirements.txt file with the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose a port (if your app runs on a specific port, e.g., 8000)
EXPOSE 8000

# Step 6: Command to run your application
CMD ["python", "app.py"]