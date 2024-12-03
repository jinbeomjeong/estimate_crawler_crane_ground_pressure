FROM python:3.10-slim
LABEL authors="jinbeom"

WORKDIR /app

RUN apt update
RUN apt install -y build-essential

RUN python -m pip install -U pip
RUN python -m pip install -U setuptools
RUN python -m pip install -U Wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
