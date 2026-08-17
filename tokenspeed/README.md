\# TokenSpeed



\[TokenSpeed](https://lightseek.org/tokenspeed/) is an OpenAI-compatible inference

engine with hybrid linear-attention support and MTP speculative decoding.



There is no official serving image. Build from `lightseekorg/tokenspeed-runner`,

then enable this backend from GPUStack's Community marketplace.



\## Hardware



| GPUStack version | GPU arch | Attention backend | Notes |

|---|---|---|---|

| `0.1.3-sm90` | Hopper (H20 / H200, sm\_90) | `fa3` | Default `tokenspeed-kernel` is Blackwell-only; Hopper must build with `TOKENSPEED\_CUDA\_ARCH=90a` |

| `0.1.3-sm100` | Blackwell (B200 / B300) | `trtllm` | `trtllm` aborts on Hopper with `TllmGenFmhaRunner: Unsupported architecture` |



Pick the version that matches the worker GPU. Mixing them will fail at runtime.



\## Build the image



The upstream `lightseekorg/tokenspeed-runner` image is a development base: CUDA/PyTorch

and prebuilt kernel wheels, but not TokenSpeed source.



```bash

docker run -itd --gpus all --name tokenspeed lightseekorg/tokenspeed-runner:latest /bin/bash

docker exec -it tokenspeed bash



export PIP\_BREAK\_SYSTEM\_PACKAGES=1

git clone https://github.com/lightseekorg/tokenspeed.git /opt/tokenspeed

cd /opt/tokenspeed

pip install -e "./python" --no-build-isolation



\# Blackwell (default kernel arch list is 100a/103a):

pip install -e tokenspeed-kernel/python/ --no-build-isolation



\# Hopper instead:

\# TOKENSPEED\_CUDA\_ARCH=90a pip install -e tokenspeed-kernel/python/ --no-build-isolation



pip install -e tokenspeed-scheduler/

chmod -R a+rX /opt/tokenspeed

