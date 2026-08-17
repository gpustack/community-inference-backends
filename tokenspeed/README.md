# TokenSpeed

## Overview

[**TokenSpeed**](https://lightseek.org/tokenspeed/) is an OpenAI-compatible inference engine with hybrid linear-attention support and MTP speculative decoding.

There is no official serving image. Build one from `lightseekorg/tokenspeed-runner`, then enable this backend from GPUStack's Community marketplace.

- Source: https://github.com/lightseekorg/tokenspeed
- Health check path: `/v1/models`

---

## Hardware

| GPUStack version | GPU arch | Attention backend | Notes |
|---|---|---|---|
| `0.1.3-sm90` | Hopper (H20 / H200, sm_90) | `fa3` | Default `tokenspeed-kernel` is Blackwell-only; Hopper must build with `TOKENSPEED_CUDA_ARCH=90a` |
| `0.1.3-sm100` | Blackwell (B200 / B300) | `trtllm` | `trtllm` aborts on Hopper with `TllmGenFmhaRunner: Unsupported architecture` |

Pick the version that matches the worker GPU. Mixing them will fail at runtime.

---

## Build the Image

The upstream `lightseekorg/tokenspeed-runner` image is a development base: it ships CUDA/PyTorch and prebuilt kernel wheels, but not the TokenSpeed source.

Start a container from the base image:

```bash
docker run -itd --gpus all --name tokenspeed lightseekorg/tokenspeed-runner:latest /bin/bash
docker exec -it tokenspeed bash
```

Inside the container, install TokenSpeed:

```bash
export PIP_BREAK_SYSTEM_PACKAGES=1

git clone https://github.com/lightseekorg/tokenspeed.git /opt/tokenspeed
cd /opt/tokenspeed
pip install -e "./python" --no-build-isolation

# Blackwell (default kernel arch list is 100a/103a):
pip install -e tokenspeed-kernel/python/ --no-build-isolation

# Hopper instead:
# TOKENSPEED_CUDA_ARCH=90a pip install -e tokenspeed-kernel/python/ --no-build-isolation

pip install -e tokenspeed-scheduler/
chmod -R a+rX /opt/tokenspeed
```

GPUStack runs the container as a non-root user, so prepare a writable `HOME` and kernel cache directories matching `default_env` in [`spec.yaml`](./spec.yaml):

```bash
mkdir -p /opt/ts-home /opt/ts-cache/flashinfer /opt/ts-cache/triton
chmod -R a+rwX /opt/ts-home /opt/ts-cache
```

Then commit the container and push it to a registry your GPUStack workers can pull from:

```bash
docker commit tokenspeed <your-registry>/tokenspeed:0.1.3-sm90
docker push <your-registry>/tokenspeed:0.1.3-sm90
```

---

## GPUStack Backend Configuration

The registered backend definition (see [`spec.yaml`](./spec.yaml)):

```yaml
backend_name: TokenSpeed
health_check_path: /v1/models
default_version: 0.1.3-sm90
is_built_in: false
parameter_format: space
default_entrypoint: tokenspeed serve
default_run_command: >
  {{model_path}}
  --port {{port}}
  --host {{worker_ip}}
  --served-model-name {{model_name}}
  --world-size {{gpu_count}}
  --chunked-prefill-size 8192
  --max-num-seqs 128
  --disable-kvstore
default_env:
  HOME: /opt/ts-home
  FLASHINFER_CACHE_DIR: /opt/ts-cache/flashinfer
  TRITON_CACHE_DIR: /opt/ts-cache/triton
common_parameters:
  - --gpu-memory-utilization
  - --max-model-len
version_configs:
  0.1.3-sm90:
    image_name: swr.cn-south-1.myhuaweicloud.com/gpustack/tokenspeed:0.1.3-sm90-cu130-20260815
    entrypoint: tokenspeed serve
    run_command: >
      {{model_path}}
      --port {{port}}
      --host {{worker_ip}}
      --served-model-name {{model_name}}
    custom_framework: cuda
  0.1.3-sm100:
    image_name: swr.cn-south-1.myhuaweicloud.com/gpustack/tokenspeed:0.1.3-cu130-20260815-fix
    entrypoint: tokenspeed serve
    run_command: >
      {{model_path}}
      --port {{port}}
      --host {{worker_ip}}
      --served-model-name {{model_name}}
    custom_framework: cuda
framework_index_map:
  cuda:
    - 0.1.3-sm90
    - 0.1.3-sm100
```

To use your own image, replace `image_name` under the matching entry in `version_configs`.
