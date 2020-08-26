ARG BASE_IMAGE=tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

FROM $BASE_IMAGE

RUN apt-get update -y && \
    apt-get install -y vim git tmux \
    libsm6 libxext6 libxrender-dev \
    build-essential busybox

RUN busybox --install

RUN git config --global core.quotepath off && \
    git config --global core.fileMode false && \
    git config --global core.editor vim

RUN pip install \
    opencv-python==4.1.1.26 \
    scikit-image \
    tifffile \
    tqdm==4.48  Keras==2.3.1 

RUN pip install pytest pylint pep8 

RUN chmod -R 777 /root

WORKDIR /app

ENV PYTHONPATH /app
ENV HOME /root
