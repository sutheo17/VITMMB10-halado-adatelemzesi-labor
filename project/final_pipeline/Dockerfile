# Start from NVIDIA’s CUDA 12.2 runtime (Ubuntu 22.04 base)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Rendszerkönyvtárak
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Conda telepítése
ARG MINICONDA_VERSION=latest
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"
SHELL ["/bin/bash", "-c"]
WORKDIR /work

# YOLOv5 klónozása
RUN mkdir -p /work/external && \
    git clone --depth 1 https://github.com/ultralytics/yolov5.git /work/external/yolov5

# Függőségek másolása és telepítése
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# A modellek beépítése a konténerbe (ÚJ SOR!)
COPY models/ /work/models/

# FONTOS: A main.py bemásolása a konténerbe
COPY main.py .

# A konténer alapértelmezett feladata a main.py futtatása lesz
ENTRYPOINT ["python", "main.py"]