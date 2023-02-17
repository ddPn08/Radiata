<div><img src="./docs/images/readme-top.png" /></div>

<div align="center">Lsmith is a fast StableDiffusionWebUI using high-speed inference technology with TensorRT</div>

---


- [Benchmark](#benchmark)
- [Screenshots](#screenshots)
- [Installation](#installation)
  - [Docker (All platform) | Easy](#docker-all-platform--easy)
    - [Customization](#customization)
  - [Linux | Difficult](#linux--difficult)
    - [requirements](#requirements)
  - [Windows | Difficult](#windows--difficult)
    - [requirements](#requirements-1)
- [Usage](#usage)
  - [Building the TensorRT engine](#building-the-tensorrt-engine)
  - [Generate images](#generate-images)

---

# Benchmark

![benchmark](./docs/images/readme-benchmark.png)

# Screenshots

- Batch generation

![lemons](./docs/images/readme-sample-screenshot-01.png)

- img2img support

![img2img](./docs/images/readme-sample-screenshot-img2img-01.png)

# Installation

## Docker (All platform) | Easy

1. Clone repository

```sh
git clone https://github.com/ddPn08/Lsmith.git
cd Lsmith
git submodule update --init --recursive
```

2. Launch using Docker compose

```sh
docker-compose up --build
```

Data such as models and output images are saved in the `docker-data` directory.

### Customization

There are two types of Dockerfile.

|                 |                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------------------- |
| Dockerfile.full | Build the TensorRT plugin. The build can take tens of minutes.                                      |
| Dockerfile.lite | Download the pre-built TensorRT plugin from Github Releases. Build times are significantly reduced. |

You can change the Dockerfile to use by changing the value of `services.lsmith.build.dockerfile` in docker-compose.yml.
By default it uses `Dockerfile.lite`.

## Linux | Difficult

### requirements

- python 3.10
- pip
- CUDA
- cuDNN < 8.6.0
- TensorRT 8.5.x

1. Clone Lsmith repository

```sh
git clone https://github.com/ddPn08/Lsmith.git
cd Lsmith
git submodule update --init --recursive
```

2. Enter the repository directory.

```sh
cd Lsmith
```

3. Run launch.sh

```sh
ex.)
bash launch.sh --host 0.0.0.0
```

## Windows | Difficult

### requirements

- python 3.10
- pip
- CUDA
- cuDNN < 8.6.0
- TensorRT 8.5.x

1. Install nvidia gpu driver
2. Instal cuda 11.x (Click [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) for the official guide)
3. Instal cudnn 8.6.0 (Click [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) for the official guide)
4. Install tensorrt 8.5.3.1 (Click [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for the official guide)
5. Clone Lsmith repository

```sh
git clone https://github.com/ddPn08/Lsmith.git
cd Lsmith
git submodule update --init --recursive
```

5. Launch `launch-user.bat`

<br />

# Usage

Once started, access `<ip address>:<port number>` (ex `http://localhost:8000`) to open the WebUI.

First of all, we need to convert our existing diffusers model to the tensorrt engine.

## Building the TensorRT engine

1. Click on the "Engine" tab
   ![](./docs/images/readme-usage-screenshot-01.png)
2. Enter Huggingface's Diffusers model ID in `Model ID` (ex: `CompVis/stable-diffusion-v1-4`)
3. Enter your Huggingface access token in `HuggingFace Access Token` (required for some repositories).
   Access tokens can be obtained or created from [this page](https://huggingface.co/settings/tokens).
4. Click the `Build` button to start building the engine.
   - There may be some warnings during the engine build, but you can safely ignore them unless the build fails.
   - The build can take tens of minutes. For reference it takes an average of 15 minutes on the RTX3060 12GB.

## Generate images

1. Select the model in the header dropdown.
2. Click on the "Generate" tab
3. Click "Generate" button.

![](./docs/images/readme-usage-screenshot-02.png)

---

<br />

Special thanks to the technical members of the [AI 絵作り研究会](https://discord.gg/ai-art), a Japanese AI image generation community.
