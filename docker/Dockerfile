ARG BASE_IMAGE=anyscale/ray:2.43.0-py312-cu125
FROM ${BASE_IMAGE}

RUN sudo apt-get update && sudo apt-get install -y jq

RUN pip install --no-cache-dir uv
RUN echo "unset RAY_RUNTIME_ENV_HOOK" >> /home/ray/.bashrc
