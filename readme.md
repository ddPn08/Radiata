<div><img src="./docs/images/readme-top.png" /></div>

<div align="center">Lsmith is a fast StableDiffusionWebUI using high-speed inference technology with TensorRT</div>

---

1. [Benchmark](#benchmark)
2. [Installation](#installation)
    - [Docker (All platform) | Easy](#docker-all-platform--easy)
    - [Linux | Difficult](#linux--difficult)
    - [Windows | Currently  unavailable](#windows--currently-unavailable)
3. [Usage](#usage)
    - [Building the TensorRT engine](#building-the-tensorrt-engine)
    - [Generate images](#generate-images)

---

# Benchmark
![benchmark](./docs/images/readme-benchmark.png)

# Installation

## Docker (All platform) | Easy

1. Clone repository
2. Launch using Docker compose
```sh
docker compose up
```

## Linux | Difficult
### requirements
- node.js (recommended version is 18)
- pnpm
- Python 3.10
- pip
- CUDA
- cuDNN < 8.6.0
- TensorRT 8.5.x

1. Follow the instructions on [this](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion#build-tensorrt-plugins-library) page to build TensorRT OSS and get `libnvinfer_plugin.so`.
2. Clone Lsmith repository
```sh
git clone https://github.com/ddPn08/Lsmith.git
cd Lsmith
git submodule update --init --recursive
```
3. Enter the repository directory.
```sh
cd Lsmith
```
5. Enter frontend directory and build frontend
```sh
cd frontend
pnpm i
pnpm build --out-dir ../dist
```
4. Run launch.sh with the path to libnvinfer_plugin.so in the LD_PRELOAD variable.
```sh
ex.)
LD_PRELOAD="/lib/src/TensorRT/build/out/libnvinfer_plugin.so.8" bash launch.sh --host 0.0.0.0
```

## Windows | Currently unavailable...
We are looking for a way to do that.
Use [Docker](#docker-all-platform--easy) instead for now.

<br />

# Usage
Once started, access `<ip address>:<port number>` (ex `http://localhost:8000`) to open the WebUI.

First of all, we need to convert our existing diffusers model to the tensorrt engine.

## Building the TensorRT engine
1. Click on the "engine" tab
![](./docs/images/readme-usage-screenshot-01.png)
2. Enter Huggingface's Diffusers model ID in `Model ID` (ex: `CompVis/stable-diffusion-v1-4`)
3. Enter your Huggingface access token in `HuggingFace Access Token` (required for some repositories).
Access tokens can be obtained or created from [this page](https://huggingface.co/settings/tokens).
4. Click the `Build` button to start building the engine.
    - There may be some warnings during the engine build, but you can safely ignore them unless the build fails.
    - The build can take tens of minutes. For reference it takes an average of 15 minutes on the RTX3060 12GB.

## Generate images
1. Select the model in the header dropdown.
2. Click on the "txt2img" tab
3. Click "Generate" button.

![](./docs/images/readme-usage-screenshot-02.png)

---

<br />

Special thanks to the technical members of the [AI絵作り研究会](https://discord.gg/ai-art), a Japanese AI image generation community.