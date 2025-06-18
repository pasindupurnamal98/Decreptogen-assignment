# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install pip, set up dependencies, and explicitly include necessary packages
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
       streamlit \
       pandas \
       numpy \
       joblib \
       pickle5 \
       plotly \
       scikit-learn \
    && pip install --no-cache-dir -r requirements.txt

COPY best_personality_model_pipeline.pkl .
COPY comprehensive_model_info.pkl .

# Copy the entire application into the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Set Streamlit to run on 0.0.0.0 so it is accessible externally
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application
CMD ["streamlit", "run", "frontend.py"]


