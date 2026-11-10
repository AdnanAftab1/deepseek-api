# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent straight to terminal (useful for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# 1. Create the virtual environment
RUN python -m venv /opt/venv

# 2. Tell Docker to use the venv's Python and pip by default
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy requirements and install into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app will run on
EXPOSE 8000

# Command to run the application using Gunicorn
# Because we updated the PATH, gunicorn is now found inside /opt/venv/bin/
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "server:app"]