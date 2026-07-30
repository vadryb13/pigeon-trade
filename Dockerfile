FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md alembic.ini ./
COPY aqr/ aqr/
COPY alembic/ alembic/

RUN pip install --no-cache-dir -e ".[llm,embeddings,data,screener]" python-multipart \
    && pip install --no-cache-dir t-tech-investments \
        --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

RUN mkdir -p /root/.aqr

EXPOSE 8000

CMD ["uvicorn", "aqr.main:app", "--host", "0.0.0.0", "--port", "8000"]
