# Windows

::: warning
This is much more difficult than the Docker installation method. Please use the Docker installation method if possible.
:::

## Requirements

- `Node.js` >= 18
- `Pnpm`
- `Python` >= 3.10
- `Pip`
- `CUDA`
- `cuDNN` < 8.6.0
- `TensorRT` 8.5.x

---

1. Install [Nvidia GPU driver](https://www.nvidia.com/download/index.aspx)
2. Instal [cuda 11.x](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
3. Instal [cudnn 8.6.0](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
4. Install [tensorrt 8.5.3.1](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

## Installation

1. Clone Lsmith repository

```sh
git clone https://github.com/ddPn08/Lsmith.git
```

2. Update the submodules

```sh
cd Lsmith
git submodule update --init --recursive
```

5. Enter frontend directory and build frontend

```sh
cd frontend
pnpm i
pnpm build --out-dir ../dist
```

6. Return to the root directory and run the application

```powershell
cd ..
.\launch-user.bat
```
