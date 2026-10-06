# Constrained benchmark environment (P4, docs/DECISIONS.md K-047). Build context: the docker/ folder only, so
# neither the datasets nor the repository are copied into the image; the repository is mounted read-only at run
# time and results/ read-write (docker/run_benchmark.ps1). CPU and memory limits are set by `docker run`.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-benchmark.txt /tmp/requirements-benchmark.txt
RUN pip install -r /tmp/requirements-benchmark.txt

WORKDIR /work
ENTRYPOINT ["python", "-m", "src.benchmark"]
