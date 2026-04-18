FROM locustio/locust

WORKDIR /mnt/locust
COPY requirements.txt .
COPY src/ .

RUN pip install --no-cache-dir -r requirements.txt
