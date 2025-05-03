ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS base

# It is more convenient for users to run as root
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install uv and python
ARG UV_VERSION=0.7.2
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0

FROM base AS hermetic

WORKDIR /opt/nemo-rl

# First copy only the dependency files
COPY --chown=ray --chmod=755 pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV VIRTUAL_ENV=/opt/nemo_rl_venv

# Create and activate virtual environment
RUN <<"EOF"
uv venv /opt/nemo_rl_venv
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

uv sync --locked --extra vllm --no-install-project
uv sync --locked --all-groups --no-install-project
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
