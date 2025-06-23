# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install setuptools==68.0.0 wheel \
    && pip install -r requirements.txt

# Expose the port your app will run on
EXPOSE 5000

# Start your app with Gunicorn
CMD ["gunicorn", "flowstate_ai:app", "--bind", "0.0.0.0:5000"]
