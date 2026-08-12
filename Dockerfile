FROM python:3.13

WORKDIR /usr/local/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./src

WORKDIR /usr/local/app/src

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]

CMD ["python", "api.py"]
