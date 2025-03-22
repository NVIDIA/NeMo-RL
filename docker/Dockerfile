ARG BASE_IMAGE=anyscale/ray:2.43.0-py312-cu125
FROM ${BASE_IMAGE} AS base
# base is just ray + uv with minimal installs so it is a very lightweight container

# It is more convenient for users to run as root
USER root

RUN apt-get update && sudo apt-get install -y jq

RUN pip install uv
RUN echo "unset RAY_RUNTIME_ENV_HOOK" >> /home/ray/.bashrc
# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0

RUN chmod 755 /home/ray/.cache

FROM base AS deps

WORKDIR /opt/reinforcer
COPY pyproject.toml .
RUN uv venv .venv
# uv sync has a more reliable resolver than simple uv pip install which can fail
RUN uv sync --extra test --extra dev --extra docs --no-install-project

FROM base as hermetic
# hermetic creates a virtual environment with the default dependencies pre-installed for convenience

ENV VIRTUAL_ENV=/opt/reinforcer/.venv
ENV PATH="/opt/reinforcer/.venv/bin:$PATH"

WORKDIR /opt/reinforcer

# The ray images automatically activate the anaconda venv. We will
# comment this out of the .bashrc to give the same UX between docker
# and other clusters like slurm.
RUN <<"EOF"
cp ~/.bashrc ~/.bashrc.backup  # backup existing .bashrc

# Comment out the conda initialize block
sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</ { /^[^#]/ s/^/# / }' ~/.bashrc

# Comment out any line that explicitly exports the anaconda3 PATH
sed -i '/export PATH=\$HOME\/anaconda3\/bin:\$PATH/ s/^/# /' ~/.bashrc
EOF

COPY --from=deps --chown=ray --chmod=755 /opt/reinforcer/.venv /opt/reinforcer/.venv
COPY --chown=ray --chmod=755 . /opt/reinforcer
RUN uv pip install --no-deps --editable .
