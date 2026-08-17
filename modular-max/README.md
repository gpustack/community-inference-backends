# Modular MAX community backend

[Modular MAX](https://docs.modular.com/intro) compiles and serves generative AI models across CPUs and NVIDIA or AMD GPUs. This backend runs the official MAX 26.4.0 containers and enables both the OpenAI-compatible endpoints and the Open Responses endpoint.

## Backend versions

| GPUStack version | Official image | Device |
| --- | --- | --- |
| `26.4.0-cuda` | `modular/max-nvidia-full:26.4.0` | NVIDIA CUDA GPU |
| `26.4.0-rocm` | `modular/max-amd:26.4.0` | AMD ROCm GPU |
| `26.4.0-cpu` | `modular/max-nvidia-base:26.4.0` | CPU |

The CUDA configuration is the default. The first deployment can remain in `Starting` for several minutes while MAX downloads weights and compiles the model.

<!-- keeping draft until apple support proven -->

Apple GPU execution is not included. Community backends currently run Linux containers, while Apple acceleration requires a native macOS MAX package and Metal device access. A container configuration marked as `mps` would therefore advertise hardware it cannot reach.

## Model sources

GPUStack resolves the selected model before starting a community backend and passes MAX an absolute local path. Consequently, this backend works with models imported into GPUStack from:

- Hugging Face;
- ModelScope; and
- a local filesystem path.

MAX itself accepts a Hugging Face repository ID or local path. It does not document a native ModelScope resolver; ModelScope works here because GPUStack downloads the repository first and presents it to MAX as a local directory. The staged repository must still use a layout and architecture supported by MAX.

Prefer importing the complete repository. When GPUStack's optional file selector matches one file, `{{model_path}}` is that file rather than its containing directory. MAX still needs the repository's `config.json`, tokenizer, and any pipeline component files. In particular, a standalone GGUF file is not sufficient; use a complete MAX-compatible repository or a local directory containing the configuration, tokenizer, and selected GGUF weight.

Private or gated Hugging Face models require accepting the model license and supplying `HF_TOKEN` as a deployment environment variable.

## Models and API routes

MAX's [supported-model matrix](https://docs.modular.com/models/) is release-sensitive and is the source of truth for architectures, modalities, encodings, and multi-GPU support. Compatible fine-tunes can work when their `config.json` names an architecture registered by MAX.

The backend enables `MAX_SERVE_API_TYPES='["openai","responses"]'`. The useful route depends on the task resolved for the selected model:

| Model task | Route |
| --- | --- |
| Text generation and multimodal text output | `/v1/chat/completions` or `/v1/completions` |
| Embeddings | `/v1/embeddings` |
| Image and video generation | `/v1/responses` |
| Readiness | `/v1/health` |
| Model metadata | `/v1/models` |

If automatic task detection is ambiguous, add `--task` in GPUStack's backend parameters. An API route being registered does not guarantee that it is valid for every deployed model.

## Precision and quantization

MAX loads SafeTensors and GGUF weights. Supported encodings include combinations of FP32, BF16, GGUF Q4/Q6, FP8, NVFP4, and GPTQ, but each architecture supports only the encodings listed in the model matrix.

Useful deployment parameters include:

- `--weight-path` selects an explicit weight file or repository;
- `--quantization-encoding` selects an encoding already present in the model repository;
- `--model-override` selects component-specific settings, including encodings in multi-component pipelines; and
- `--kv-cache-format` independently selects FP32, BF16, or FP8 KV-cache storage.

`--quantization-encoding` is not a general online converter. For example, selecting `float8_e4m3fn` does not turn arbitrary BF16 SafeTensors into an FP8 checkpoint. Use weights that were produced for the desired encoding and supported by the architecture. GGUF encodings are normally detected automatically.

MAX 26.4 runs GGUF Q4 and Q6 kernels on CPU. If a CUDA deployment resolves one of those encodings, MAX switches the model device to CPU; select the CPU backend version so GPUStack schedules and accounts for the workload correctly. GGUF conversion details also matter: a third-party file with an otherwise supported label can contain unsupported tensor subtypes. Modular's example GGUF repository is the safest compatibility reference.

Hardware support is also precision-specific. In particular, native NVFP4 matrix multiplication requires a Blackwell-class NVIDIA GPU; an Ampere GPU can run MAX but cannot be assumed to run an NVFP4 checkpoint unless that model implementation documents a fallback.

## Validation notes

The 26.4.0 configurations were exercised through GPUStack 2.2.3 on an NVIDIA RTX 3090 and a 64 GiB x86-64 host:

| Source / model | Device and encoding | Result |
| --- | --- | --- |
| Hugging Face `LiquidAI/LFM2.5-350M` | CUDA, BF16 | `/v1/health`, model metadata, and chat completion succeeded |
| ModelScope `sentence-transformers/all-MiniLM-L6-v2` | CPU, FP32 | two 384-dimensional embeddings succeeded |
| GPUStack local directory with LFM2.5 | CUDA, BF16 | health, chat, and streaming chat completion succeeded |
| `Qwen/Qwen3-0.6B-FP8` | CUDA, FP8 weights | MAX selected FP8, then rejected the checkpoint's BF16 scale tensors |
| `bartowski/Llama-3.2-1B-Instruct-GGUF` | Q4_0 | MAX selected Q4_0 and CPU, then rejected a Q4_1 tensor subtype in the conversion |

The negative precision cases are intentional compatibility controls: they confirm that the flags select existing encodings rather than converting weights. They do not imply that every FP8 or Q4 model is unsupported. Use the exact architecture/encoding combinations and example repositories in `max list` or the supported-model matrix.

## Generative image and video models

The public MAX 26.4 model matrix includes FLUX.2 image generation and Wan 2.1/2.2 video generation pipelines. These models use `/v1/responses` and generally require substantially more memory than small language or embedding models.

For scale, the Hugging Face repositories total about 23.7 GB for FLUX.2 Klein 4B BF16, 2.5 GB for its NVFP4 variant, 34.2 GB for Wan 2.2 TI2V 5B, and 126.2 GB for Wan 2.2 T2V A14B. The BF16 FLUX checkpoint leaves no safe runtime headroom on a 24 GB GPU, while the smaller NVFP4 checkpoint requires Blackwell-class hardware. The Wan repositories exceed that test system's capacity, and Modular documents video generation as GPU-required, so 64 GB of CPU RAM is not a substitute for GPU memory in this case.

Modular's model catalog separately advertises LTX-2.3 as a 22B NVFP4 model with a roughly 21.7 GB Hugging Face repository. At the time this backend was added, LTX-2.3 was not present in the public MAX supported-model matrix or open-source architecture registry. Its NVFP4 encoding also does not target an RTX 3090. Do not assume that the catalog model can be started with the public `max serve` container until Modular publishes the corresponding architecture or self-hosted invocation.

## Parameters

GPUStack accepts additional MAX CLI options in **Backend Parameters**. Common examples are:

```text
--task text_generation
--device-memory-utilization 0.8
--quantization-encoding q4_k
--kv-cache-format bfloat16
```

Use `--pretty-print-config` while diagnosing a deployment to record the resolved model, task, devices, and encoding. See the [`max serve` reference](https://docs.modular.com/cli/serve/) for the complete and version-specific option list.

## Licensing

The Modular source repository is Apache-2.0 with LLVM exceptions, while downloading or using the MAX and Mojo SDK or container is governed by the [Modular Community License](https://www.modular.com/legal/community). This backend definition references Modular's official container images; it does not redistribute the MAX SDK.

The Community License is free but is not an open-source software license. Non-production use has no device limit. For commercial production, it permits unlimited x86, ARM, and NVIDIA PTX devices, but limits other accelerator types to eight devices in aggregate unless Modular grants a larger license. An entity starting a commercial production deployment must notify Modular within 30 days, permit identification as an SDK user, and display "Powered by Modular." The license also restricts redistribution, sublicensing, reverse engineering, removal of notices, and unsupported hardware use, and permits collection of usage telemetry.

The included MAX logo is an approved asset from Modular's [trademark policy](https://www.modular.com/legal/trademark). MAX® and Mojo® are trademarks of Modular, Inc. used under license. This community integration is not operated by or endorsed by Modular.

Model checkpoints retain their own licenses. Review and accept all applicable terms before deployment.
