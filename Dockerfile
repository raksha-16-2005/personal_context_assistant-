# Multi-stage: build the React frontend with Node, install Python deps in a
# builder that has a C compiler, then run from a slim final image that has
# neither Node nor build tools - only the built frontend and a venv. Cuts
# the deployed image by the full weight of build-essential + npm's own
# toolchain, neither of which the running app ever needs.

FROM node:20-slim AS frontend
WORKDIR /frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS builder
WORKDIR /app

# PyStemmer (bm25s's stemmer backend) has no manylinux wheel for every
# platform this might land on, so a C compiler has to be present to build it
# from sdist - only in this stage, never in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# torch has no CPU-only wheel on the default PyPI index - installing from it
# would pull the CUDA-bundled build (multiple GB) for a container that only
# ever runs on CPU. Installed first, pinned, from PyTorch's own CPU index;
# the requirements.txt line below then finds it already satisfied.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
COPY webapp/requirements.txt ./webapp/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r webapp/requirements.txt

FROM python:3.11-slim AS backend
WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY src/ ./src/
COPY webapp/app/ ./webapp/app/
COPY --from=frontend /frontend/dist/ ./webapp/frontend/dist/

WORKDIR /app/webapp
EXPOSE 8000

# Overridden per process group in fly.toml ([processes].app / .worker) -
# this default is only what runs if the image is started with no override,
# e.g. `docker run` for a local smoke test.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
