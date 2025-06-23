# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc ffmpeg libsndfile1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install setuptools==68.0.0 wheel
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose the port Railway or other platforms use
EXPOSE $PORT

# Command to run the app with gunicorn
CMD ["gunicorn", "flowstate_ai:app", "--bind", "0.0.0.0:$PORT"]
