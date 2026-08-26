FROM python:3.13

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts
COPY sagemaker ./sagemaker

# Dataset is mounted at runtime
RUN mkdir -p /app/data

CMD ["python", "scripts/prepare_s3.py"]