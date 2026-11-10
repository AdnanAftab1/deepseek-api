# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to terminal (useful for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container (FIXED: Changed from / to /app)
WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
# If requirements don't change, this layer will be cached
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This includes server.py, DeepSeekAPI.py, and your .env file
COPY . .

# Expose the port your Flask app will run on
EXPOSE 8000

# Command to run the application using Gunicorn
# FIXED: Changed 'your_server_file_name:app' to 'server:app'
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "server:app"]