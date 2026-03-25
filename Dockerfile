FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the project in editable mode so internal imports resolve
RUN pip install -e .

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# Corrected path to the app entry point
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]