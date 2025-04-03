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

FROM base AS hermetic

WORKDIR /opt/reinforcer
# This is less efficient as this invalidates the cache more frequently, but
# creates a smaller image. Adding reinforcer afterwards and doing
# `uv pip install --no-deps --editable .` causes a "sync" of some of the environment,
# which defeats the purpose of pre-installing.
# In the future we may optimize this: https://github.com/NVIDIA/reinforcer/issues/129
COPY --chown=ray --chmod=755 . /opt/reinforcer
RUN <<"EOF"
uv venv .venv
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

uv sync --locked --extra vllm --no-install-project
uv sync --locked --all-groups
EOF

ENV VIRTUAL_ENV=/opt/reinforcer/.venv
ENV PATH="/opt/reinforcer/.venv/bin:$PATH"

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
