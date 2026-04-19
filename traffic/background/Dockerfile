FROM locustio/locust

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /mnt/locust
COPY src/ .
COPY pyproject.toml .

RUN uv sync
