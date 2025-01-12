# Use official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . /app/

# Expose port 8000 for Django and 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Set the environment variable to tell Django to run in production mode
ENV DJANGO_SETTINGS_MODULE=stock_predictor.settings

# Define the command to run the Django and Streamlit apps (via start.sh)
CMD ["/bin/bash", "/app/start.sh"]
