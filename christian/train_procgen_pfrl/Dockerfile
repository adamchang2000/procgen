FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt install -y --no-install-recommends \
       wget \
       ca-certificates \
       locales \
       libglib2.0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       xvfb \
       ffmpeg \
       freeglut3-dev \
       curl \
       bzip2 \
       build-essential \
       software-properties-common \
    && rm -rf /var/cache/*

RUN apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       g++  \
       git  \
       cmake \
       zlib1g-dev \
       libjpeg-dev \
       xorg-dev \
       libboost-all-dev \
       libsdl2-dev \
       swig \
       libopenblas-base  \
       cython3  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# RUN conda init && /root/.bashrc
# RUN conda activate procgen && cd /root/procgen/ && pip install -e .

