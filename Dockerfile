FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts

# CPU-only PyTorch keeps the image small.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Artifacts come from `python -m rtrec.pipeline --out artifacts`, run before building.
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "rtrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
