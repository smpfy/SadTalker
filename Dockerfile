FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3.8 python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR SadTalker/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY checkpoints/ checkpoints/
COPY gfpgan/ gfpgan/

COPY src/ src/
COPY inference.py inference.py
COPY app_sadtalker.py app_sadtalker.py