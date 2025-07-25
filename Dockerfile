# Use an official lightweight Python image
FROM python:3.11-slim-bullseye


# Set working directory
WORKDIR /app

# Copy the code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8100

# Command to run the API
CMD ["uvicorn", "title_similarity_service:app", "--host", "0.0.0.0", "--port", "8100"]
