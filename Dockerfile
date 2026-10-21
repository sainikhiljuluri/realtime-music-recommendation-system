FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACT_DIR=/app/artifacts

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# libgomp1: LightGBM's shared library links against the system OpenMP runtime. The slim base image
# does not include it, and the container exits at import without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small. The API runs as an unprivileged user.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu . \
 && useradd --system --uid 10001 --no-create-home app

# Artifacts are mounted read-only at /app/artifacts (see docker-compose.yml). They are not baked
# into the image, so the same image serves any trained artifact directory.
USER app
EXPOSE 8000

# Readiness, not liveness: the container is healthy only when the model is loaded and self-checked.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"

# --timeout-graceful-shutdown: on SIGTERM, stop accepting connections and finish in-flight requests
# for up to 20 seconds before exiting.
CMD ["uvicorn", "rtrec.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
