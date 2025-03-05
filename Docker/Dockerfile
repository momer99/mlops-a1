# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt from the parent directory
COPY ../src/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application files from the Docker directory
COPY ../src /app

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable (optional)
ENV FLASK_APP=app.py

# Command to run the application
CMD ["python", "app.py"]
