FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip build-essential swig curl ca-certificates git \
    ffmpeg zstd libegl1 libgl1 libgles2 libglfw3 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libasound2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/fast-jepa
COPY requirements-blackwell.lock ./
COPY scripts/install.sh scripts/install.sh
COPY tools/prefetch_cuda.py tools/prefetch_cuda.py
RUN bash scripts/install.sh
ENV PATH=/opt/fast-jepa/.venv/bin:$PATH \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    STABLEWM_HOME=/data
COPY . .
CMD ["bash"]
