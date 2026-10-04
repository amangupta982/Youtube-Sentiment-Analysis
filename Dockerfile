FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app

# Install system dependencies required for LightGBM and other ML libraries
RUN apt-get update && apt-get install -y libgomp1 gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app runs on
EXPOSE 5001

# Command to run the application
CMD ["python", "flask_app/app.py"]
