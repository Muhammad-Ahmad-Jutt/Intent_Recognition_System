FROM python:3.13-slim

WORKDIR /opt/program

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip setuptools wheel

COPY inference/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY inference/inference.py .

EXPOSE 8080

ENTRYPOINT ["python", "inference.py"]