FROM python:3.12

WORKDIR /src
COPY requirements.txt .
COPY src/ .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]