FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY gridcast ./gridcast
RUN pip install ".[models,deep,serve]"

COPY dashboard ./dashboard

EXPOSE 8000
CMD ["gridcast", "serve", "--host", "0.0.0.0", "--port", "8000"]
