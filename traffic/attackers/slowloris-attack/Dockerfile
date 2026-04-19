FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src
COPY src/ .
COPY pyproject.toml .

RUN uv sync

CMD [ "uv", "run", "main.py" ]