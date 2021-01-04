# This docker sets up an environment with RedHat7 and Python3
# that runs the sparsezoo python module

# Install OS
FROM registry.access.redhat.com/ubi7/ubi:latest
ARG gid
RUN yum update -y

# Install misc packages
RUN yum install -y \
    git \
    vim \
    wget

RUN yum install -y \
    scl-utils \
    rh-python36 \
    rh-python36-python-pip \
    rh-python36-python-devel \
    && scl enable rh-python36 bash \
    && export PATH=/opt/rh/rh-python36/root/usr/bin:$PATH

# set up pip
RUN /opt/rh/rh-python36/root/usr/bin/pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# Give users access to python3, pip3 and jupyter
RUN ln -s /opt/rh/rh-python36/root/usr/lib64/libpython3.6m.so.rh-python36-1.0 /usr/lib64/libpython3.6m.so.1.0
RUN ln -s /opt/rh/rh-python36/root/usr/bin/python3 /usr/bin/python3
RUN ln -s /opt/rh/rh-python36/root/usr/bin/pip3 /usr/bin/pip3
RUN ln -s /opt/rh/rh-python36/root/usr/bin/jupyter /usr/bin/jupyter

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
