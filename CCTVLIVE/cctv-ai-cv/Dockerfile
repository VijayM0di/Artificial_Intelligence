FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get -y upgrade && apt install -y \
    apt-utils \
    build-essential \
    pkg-config \
    libcurl4-openssl-dev \
    nano \
    gcc \
    curl \
    unixodbc-dev \
    unixodbc \
    ffmpeg \
    supervisor \
    libopencv-highgui-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 5500
ENV PYTHONPATH=/app

ENTRYPOINT service supervisor start && python prod_camera_starter_service.py && python ai_services_api.py && /bin/bash
