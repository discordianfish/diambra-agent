FROM docker.io/python:3.10

RUN apt-get -qy update && \
  apt-get -qy install libgl1 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
