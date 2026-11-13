FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tini && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home --uid 10001 trader && mkdir -p /app/data /app/logs /app/reports && chown -R trader /app
USER trader
ENTRYPOINT ["/usr/bin/tini", "--"]
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 CMD python scripts/healthcheck.py || exit 1
CMD ["python", "main.py", "run"]
