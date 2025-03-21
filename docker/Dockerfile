ARG BASE_IMAGE=anyscale/ray:2.43.0-py312-cu125
FROM ${BASE_IMAGE}

WORKDIR /opt/NeMo-Reinforcer

RUN sudo apt-get update && sudo apt-get install -y jq

RUN echo "unset RAY_RUNTIME_ENV_HOOK" >> /home/ray/.bashrc

COPY pyproject.toml .

RUN uv pip install -r pyproject.toml

COPY . .

RUN uv pip install -e .[dev,test]
