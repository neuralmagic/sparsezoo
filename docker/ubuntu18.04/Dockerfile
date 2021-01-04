# This docker sets up an environment with Ubuntu 18.04 and Python3
# that runs the sparsezoo python module

# Install OS
FROM ubuntu:18.04
RUN apt update
RUN apt install -y \
    bash \
    build-essential \
    git \
    curl

# Install python3 packages
RUN apt install -y \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists

# set up pip
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# Create user
RUN if [ -z "$gid" ] ; then groupadd nm_group ; else groupadd -o --gid $gid nm_group ; fi
RUN useradd -m nm_user -G nm_group

# Install sparsezoo
COPY . /home/nm_user/sparsezoo
RUN python3 -m pip install /home/nm_user/sparsezoo/

# Finish setup
RUN chown -R nm_user /home/nm_user
USER nm_user
WORKDIR /home/nm_user
