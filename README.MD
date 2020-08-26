# PIEA

## Setup 

1. Download weights

```bash
mkdir -p weights

curl -o weights/mobilenet_weights.h5 https://github.com/lshdan/piea/releases/download/1.0/mobilenet_weights.h5

curl -o weights/inception_resnet_weights.h5 https://github.com/lshdan/piea/releases/download/1.0/inception_resnet_weights.h5
```

2. Build environment ([docker](https://docs.docker.com/engine/install/ubuntu/) is required)

```bash
make build
```

## Run the demo

- CPU version:

```bash

make run-piea-cpu 

```

- GPU version: ([docker with nvidia runtime](https://github.com/NVIDIA/nvidia-docker) required)

```bash

make run-piea-gpu 

```
