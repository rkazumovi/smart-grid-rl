# Dockerfile for the Smart Grid RL forecasting/policy API (src/api/).
#
# This is NOT a training image -- it only ever loads already-trained checkpoints from
# outputs/ and runs inference, so it deliberately installs the CPU build of torch (a
# fraction of the size of the CUDA build, and this project's models are small enough --
# tens of thousands of parameters -- that CPU inference is fast) rather than trying to
# pass a GPU through to a container, which most deployment targets (a laptop's Docker
# Desktop, most managed container platforms) don't support anyway.
#
# Multi-stage build: the "builder" stage has the C/Fortran toolchain pandapower's SciPy
# dependency needs to compile from source on some platforms; the final stage copies only
# the installed Python packages out of it, not the compiler toolchain itself, keeping the
# shipped image smaller.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-api.txt .

# CPU-only torch wheel from PyTorch's own index -- the default PyPI wheel pulls in the
# much larger CUDA runtime, which this inference-only image never uses.
RUN pip install --no-cache-dir --user \
        torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user \
        torch_geometric==2.8.0.post1 \
    && pip install --no-cache-dir --user -r requirements-api.txt

FROM python:3.11-slim

# libgomp1: OpenMP runtime torch's CPU kernels link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY src/ src/
COPY data/processed/ data/processed/
COPY outputs/ outputs/

RUN chown -R appuser:appuser /app /home/appuser/.local
USER appuser

# HOME is set explicitly rather than left to whatever the container runtime infers for a
# non-login process -- Python's user-site-packages mechanism (what makes the --user
# installs above importable) resolves relative to $HOME, and pip's own installed console
# scripts (uvicorn) live under $HOME/.local/bin.
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Shell form (not the usual exec-form JSON array) specifically so ${LOG_LEVEL:-info} gets
# expanded -- an exec-form CMD is passed straight to the syscall with no shell involved,
# so that substitution would otherwise be sent to uvicorn as a literal, unexpanded string
# and it would silently fall back to its own default instead of honoring LOG_LEVEL (see
# k8s/configmap.yaml, which sets this env var).
CMD uvicorn main:app --app-dir src/api --host 0.0.0.0 --port 8000 --log-level ${LOG_LEVEL:-info}
