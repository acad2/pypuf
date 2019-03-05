FROM python:3.7-alpine

RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
VOLUME . /app/