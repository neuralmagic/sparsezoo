# This docker sets up an environment with CentOS8 and Python3
# that runs the sparsezoo python module

# Install OS
FROM centos:8
ARG gid
RUN yum update -y

# Install misc packages
RUN yum install -y \
    git \
    vim \
    wget

# Install python3 packages
RUN yum install -y \
    python36 \
    python3-pip \
    python3-devel

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
