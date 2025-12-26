FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install libgl1 for cv2
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY . .

EXPOSE 8000 8501
