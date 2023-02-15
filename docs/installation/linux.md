# Linux

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

## Installation

1. Clone Lsmith repository

```sh
git clone https://github.com/ddPn08/Lsmith.git
```

2. Install submodules

```sh
cd Lsmith
git submodule update --init --recursive
```

3. Enter the repository directory.

```sh
cd Lsmith
```

4. Enter frontend directory and build frontend

```sh
cd frontend
pnpm i
pnpm build --out-dir ../dist
```

5. Run launch.sh with the path to libnvinfer_plugin.so in the LD_PRELOAD variable.

```sh
bash launch.sh --host 0.0.0.0
```
