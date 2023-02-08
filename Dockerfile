FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as tensorrt

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y \
    && apt install software-properties-common -y \
    && add-apt-repository --yes ppa:deadsnakes/ppa

RUN apt update && apt install build-essential curl git-core tensorrt tensorrt-dev tensorrt-libs -y

RUN git clone https://github.com/NVIDIA/TensorRT /TensorRT && cd /TensorRT && git submodule update --init --recursive

WORKDIR /TensorRT

RUN curl https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh -L -o ./install_cmake \
    && chmod +x ./install_cmake \
    && mkdir -p /opt/cmake \
    && ./install_cmake --skip-license --prefix="/opt/cmake" \
    && ln -s /opt/cmake/bin/* /usr/bin \
    && ls -al /opt/cmake

RUN mkdir -p build && cd build \
    && cmake .. -DTRT_OUT_DIR=$PWD/out \
    && cd plugin \
    && make -j$(nproc)


FROM node:18.14.0-alpine3.17 as frontend

COPY . /Lsmith

RUN apk update && apk add git && npm i -g pnpm

WORKDIR /Lsmith/frontend

ENV VITE_API_BASE_PATH=""

RUN pnpm i && pnpm build


FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as main

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y \
    && apt install software-properties-common -y \
    && add-apt-repository --yes ppa:deadsnakes/ppa

RUN apt update && apt install curl python3.10 python3.10-venv git-core tensorrt=8.5.3.1-1+cuda11.8 -y

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10

COPY . /app

WORKDIR /app

RUN git submodule update --init --recursive

COPY --from=tensorrt /TensorRT/build/out/libnvinfer_plugin.so.8 /app/lib/trt/lib/libnvinfer_plugin.so
COPY --from=frontend /Lsmith/frontend/dist /app/dist

ENTRYPOINT [ "/usr/bin/python3.10", "-u", "/app/launch.py" ]
