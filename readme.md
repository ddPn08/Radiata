<div><img src="./docs/images/readme-top.png" /></div>

<div align="center">Lsmith is a fast StableDiffusionWebUI using high-speed inference technology with TensorRT</div>

---

# Benchmark
![benchmark](./docs/images/readme-benchmark.png)

# Usage

## Docker | Easy

1. Clone repository
2. Launch using Docker compose
```sh
docker compose up
```

## Linux | Difficult
### requirements
- Python 3.10
- pip
- CUDA
- cuDNN < 8.6.0
- TensorRT 8.5.x

1. Follow the instructions on [this](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion#build-tensorrt-plugins-library) page to build TensorRT OSS and get `libnvinfer_plugin.so`.
2. Clone Lsmith repository
```sh
git clone https://github.com/ddPn08/Lsmith.git
```
3. Enter the repository directory.
```sh
cd Lsmith
```
4. Run launch.sh with the path to libnvinfer_plugin.so in the LD_PRELOAD variable.
```sh
ex. )
LD_PRELOAD="/lib/src/TensorRT/build/out/libnvinfer_plugin.so.8" bash launch.sh --host 0.0.0.0
```

## Windows | Unavailable now...
We are looking for a way to do that.

<br />

---

<br />

Special thanks to the technical members of the [AI絵作り研究会](https://discord.gg/ai-art), a Japanese AI image generation community.