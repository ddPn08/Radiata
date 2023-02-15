# Docker installation

## Cloning the repository

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

## Customization

There are two types of Dockerfile.

| Filename        | Features                                                                                            |
| --------------- | --------------------------------------------------------------------------------------------------- |
| Dockerfile.full | Build the TensorRT plugin. The build can take tens of minutes.                                      |
| Dockerfile.lite | Download the pre-built TensorRT plugin from Github Releases. Build times are significantly reduced. |

You can change the Dockerfile to use by changing the value of `services.lsmith.build.dockerfile` in docker-compose.yml.
By default it uses `Dockerfile.lite`.
