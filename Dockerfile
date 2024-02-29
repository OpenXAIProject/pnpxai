FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3.9 \
        python3-pip

COPY . /app

WORKDIR /app

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -e .

RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html