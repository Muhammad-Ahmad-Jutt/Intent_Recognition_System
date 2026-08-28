FROM python:3.13-slim

WORKDIR /opt/ml/code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# RUN python -m pip install --upgrade pip setuptools wheel

COPY training/requirements.txt .

RUN pip install  -r requirements.txt

COPY training/train.py .

ENTRYPOINT ["python", "train.py"]