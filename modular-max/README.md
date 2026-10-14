# Modular MAX community backend

[Modular MAX](https://docs.modular.com/intro) compiles and serves generative AI models across CPUs and NVIDIA or AMD GPUs. This backend runs the official MAX 26.6.0 containers and enables both the OpenAI-compatible endpoints and the Open Responses endpoint.

## Backend versions

| GPUStack version | Official image | Framework | Platforms |
| --- | --- | --- | --- |
| `26.6.0-cuda` | `modular/max-nvidia-full:26.6.0` | `cuda` | linux/amd64, linux/arm64 |
| `26.6.0-rocm` | `modular/max-amd:26.6.0` (ROCm 7.2) | `rocm` | linux/amd64 |
| `26.6.0-cpu` | `modular/max-nvidia-base:26.6.0` | `cpu` | linux/amd64, linux/arm64 |

The CUDA configuration is the default. The first deployment can remain in `Starting` for several minutes while MAX downloads weights and compiles the model.

These are the device families MAX itself supports in Linux containers. Only the CUDA and CPU versions have been tested on real hardware for this backend (see [Validation notes](#validation-notes)). The ROCm version uses the same command line with Modular's AMD image, but it has not been run on an AMD GPU yet.

The CPU version is meant for CPU-only workers. On a worker that has GPUs, GPUStack 2.2.3 still assigns one GPU to a `26.6.0-cpu` instance and passes it into the container: MAX computes on the CPU because of `--devices cpu`, but its processes hold about 0.5 GB of CUDA context on that GPU. GPUStack's custom-backend scheduler only considers CPU-only placement when the model has CPU offloading enabled and no GPU fits.

Apple GPU execution is not included. Community backends currently run Linux containers, while Apple acceleration requires a native macOS MAX package and Metal device access. A container configuration marked as `mps` would therefore advertise hardware it cannot reach.

## Model sources

GPUStack resolves the selected model before starting a community backend and passes MAX an absolute local path. Consequently, this backend works with models imported into GPUStack from:

- Hugging Face;
- ModelScope; and
- a local filesystem path.

MAX itself accepts a Hugging Face repository ID or local path. It does not document a native ModelScope resolver; ModelScope works here because GPUStack downloads the repository first and presents it to MAX as a local directory. The staged repository must still use a layout and architecture supported by MAX.

Prefer importing the complete repository. When GPUStack's optional file selector matches one file, `{{model_path}}` is that file rather than its containing directory. MAX still needs the repository's `config.json`, tokenizer, and any pipeline component files. In particular, a standalone GGUF file is not sufficient; use a complete MAX-compatible repository or a local directory containing the configuration, tokenizer, and selected GGUF weight.

Private or gated Hugging Face models require accepting the model license and supplying `HF_TOKEN` as a deployment environment variable.

## Supported models

MAX's [supported-model matrix](https://docs.modular.com/models/) and `max list` are release-sensitive and are the source of truth for architectures, encodings, and multi-GPU support. Compatible fine-tunes can work when their `config.json` names an architecture registered by MAX. MAX 26.6.0 registers these families:

| Modality | Architectures (example repositories) |
| --- | --- |
| Text generation | Llama 3 (`meta-llama/Llama-3.1-8B-Instruct`), Qwen 2.5 / 3 / 3 MoE (`Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B-Instruct`), Gemma 3 1B, Mistral (`mistralai/Mistral-Nemo-Instruct-2407`), Phi-3/4, Granite, OLMo 1/2/3, LFM2 (`LiquidAI/LFM2.5-350M`), Mamba, Nemotron-H, GPT-OSS (`openai/gpt-oss-20b`), DeepSeek V2/V3/V3.2, GLM-5.x, MiniMax M2, Step-3.5, Hy3, Laguna, DiffusionGemma |
| Vision-language | Qwen2.5-VL, Qwen3-VL (`Qwen/Qwen3-VL-4B-Instruct`), Qwen3.5 / 3.6 / 3.8 (`Qwen/Qwen3.5-27B`), Gemma 3 4B+ and Gemma 4, Llama 4 Scout, Mistral Small 3.1, Pixtral, InternVL3, Idefics3, Kimi-VL, Kimi K2.5, Inkling |
| Embeddings | BERT (`sentence-transformers/all-MiniLM-L6-v2`), MPNet (`sentence-transformers/all-mpnet-base-v2`), Qwen3 embedding models with `--task embeddings_generation` |
| Image generation and editing | FLUX.2 dev and Klein (`black-forest-labs/FLUX.2-klein-4B`), Qwen-Image and Qwen-Image-Edit, Z-Image, Ideogram 4 |
| Video generation | Wan 2.1 / 2.2 text-to-video and image-to-video (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) |
| Audio generation | MiniMax-Music3 (`MiniMaxAI/MiniMax-Music3`) text-to-music |

Speculative decoding drafters (Eagle, Eagle3, MTP, DFlash, DSpark) are also registered; they are passed with `--speculative-method` and `--draft-model-path` rather than deployed as standalone models. Exaone was removed from the registry between 26.4 and 26.6.

## API routes

The backend enables `MAX_SERVE_API_TYPES='["openai","responses"]'`. The useful route depends on the task resolved for the selected model:

| Model task | Route |
| --- | --- |
| Text generation and multimodal text output | `/v1/chat/completions` or `/v1/completions` |
| Embeddings | `/v1/embeddings` |
| Image and video generation | `/v1/responses` |
| Audio generation | `/v1/audio/speech` (WAV) or `/v1/responses` |
| Readiness | `/v1/health` |
| Model metadata | `/v1/models` |

If automatic task detection is ambiguous, add `--task` in GPUStack's backend parameters. A registered route is not valid for every model: in 26.6.0, `/v1/responses` serves the pixel and audio pipelines, and a text model deployed with the default settings answers it with HTTP 500. Use the chat or completions routes for text models.

Requests larger than 100 MiB are rejected with HTTP 413. Set `MAX_SERVE_MAX_REQUEST_BYTES` in the deployment environment to raise the limit for large inline images or video, or `0` to disable it.

## Chat features

Tool calling and reasoning output depend on a parser for the model family. When `--tool-parser` and `--reasoning-parser` are unset, MAX applies the architecture's default parser if it declares one. In 26.6.0, Qwen3.5/3.6, Gemma 4, GLM-5.x, Kimi K2.5, MiniMax M2, Laguna, and Inkling declare defaults. Other families return tool calls and `<think>` blocks as plain text in `content`. An explicit `--tool-parser` only helps when the parser is registered for the served model: each family's parser is loaded with its own architecture (`deepseekv3` and `deepseekv3_1` come with DeepSeek V3), and only the generic `llama` parser is always present. MAX does not check the name at startup. With an unavailable parser, for example `--tool-parser qwen3_5` on `Qwen/Qwen3-0.6B`, the server starts and then answers every chat request with 400 `Unknown tool parser`. A reasoning parser also needs tokenizer support: the text-only Qwen3 architecture (`Qwen3ForCausalLM`, for example `Qwen/Qwen3-4B`) refuses to start with `--reasoning-parser qwen3_5`, so its `<think>` block always stays in `content`.

`response_format` (JSON schema) requires `--enable-structured-output`. Without the flag, MAX rejects such requests with 400. With the flag, constrained decoding did not hold in any configuration tried on the RTX 3090 with 26.6.0:

- `Qwen/Qwen3.5-4B` answered with Markdown-fenced JSON that ignored the schema, with both the `xgrammar` and `llguidance` backends and with thinking on or off. MAX logged `response_format_conformance ... outcome=invalid_json` for every such request.
- `Qwen/Qwen3-4B` crashed the model worker on the first schema request (`Generated out-of-vocabulary token_id=151680 ... (valid range: [0, 151669))`), so the instance stopped serving everyone.
- `Qwen/Qwen3-0.6B` returned JSON with the schema's keys but meaningless values.

Keep the flag off unless structured output has been checked for the specific model and GPU.

## Precision and quantization

MAX loads SafeTensors and GGUF weights. Supported encodings include combinations of FP32, FP16, BF16, GGUF Q4/Q6, FP8, FP4 (NVFP4 and MXFP4 checkpoints), and GPTQ, but each architecture supports only the encodings listed in the model matrix.

Useful deployment parameters include:

- `--weight-path` selects an explicit weight file or repository;
- `--quantization-encoding` selects an encoding already present in the model repository;
- `--model-override` selects component-specific settings, including encodings in multi-component pipelines; and
- `--kv-cache-format` independently selects FP32, BF16, or FP8 KV-cache storage.

`--quantization-encoding` is not a general online converter. For example, selecting `float8_e4m3fn` does not turn arbitrary BF16 SafeTensors into an FP8 checkpoint. Use weights that were produced for the desired encoding and supported by the architecture. GGUF encodings are normally detected automatically.

MAX 26.6 still runs GGUF Q4 and Q6 kernels on CPU only. If a CUDA deployment resolves one of those encodings, MAX logs `Encoding 'q4_k' is only supported on CPU` and moves the model to the CPU; select the CPU backend version so GPUStack schedules and accounts for the workload correctly. A GGUF repository with several files can also resolve to its BF16 file unless `--quantization-encoding` names the one you want. GGUF conversion details matter as well: a third-party file with an otherwise supported label can contain unsupported tensor subtypes. Modular's example GGUF repository is the safest compatibility reference.

Hardware support is also precision-specific. In particular, native NVFP4 matrix multiplication requires a Blackwell-class NVIDIA GPU; an Ampere GPU can run MAX but cannot be assumed to run an NVFP4 checkpoint unless that model implementation documents a fallback.

## Memory and KV cache

`--device-memory-utilization` (default `0.9`) is the fraction of free device memory MAX takes; whatever the weights leave of it becomes the KV cache. On a consumer GPU that also drives a display, the default can leave too little headroom for activations: Qwen3-4B on a 24 GB RTX 3090 ran out of CUDA memory under 64 concurrent requests at the default and was stable at `--device-memory-utilization 0.8`.

Prefix caching is on by default and reuses prefixes that are still in the GPU KV cache (`--no-enable-prefix-caching` turns it off). The public 26.6.0 containers have no host or disk KV tier. `--kv-connector-config '{"type":"rust_tiered"}'` (and its `tiered` alias) needs a Rust `kv_tier_connector` extension that is not shipped in the images, so the model worker exits with `ModuleNotFoundError: No module named 'kv_tier_connector'`. The remaining `dkv` connector needs an external block store endpoint. Once a prefix is evicted from GPU memory, MAX recomputes it.

## Validation notes

MAX 26.6.0 was run on one NVIDIA RTX 3090 (24 GB, also driving a desktop) with a 64 GiB x86-64 host, using the same images and commands as the backend versions. The ROCm version and ARM64 hosts were not tested.

| Model | Version and settings | Result |
| --- | --- | --- |
| `Qwen/Qwen3-4B` | `26.6.0-cuda`, BF16, `--device-memory-utilization 0.8` | chat and completions passed; 701 output tok/s at 64 concurrent requests (1024 in / 256 out) |
| `Qwen/Qwen3.5-4B` | `26.6.0-cuda`, BF16, `--device-memory-utilization 0.8` | chat, streaming, tool call, separate `reasoning` field, and image input passed; `response_format` returned 400 without `--enable-structured-output` and ignored the schema with it (see [Chat features](#chat-features)); `/v1/responses` returned 500 |
| `Qwen/Qwen3-Embedding-0.6B` | `26.6.0-cuda`, `--task embeddings_generation` | two 1024-dimensional embeddings |
| `sentence-transformers/all-MiniLM-L6-v2` | `26.6.0-cpu` | two 384-dimensional embeddings |
| `modularai/Llama-3.1-8B-Instruct-GGUF` | `26.6.0-cuda`, `--quantization-encoding q4_k` | MAX moved the model to CPU; chat passed |
| `Qwen/Qwen3-4B-FP8` | `26.6.0-cuda` | MAX rejected the checkpoint's BF16 scale tensors (`a_scales and b_scales dtypes must be float32`) |
| `black-forest-labs/FLUX.2-klein-4B` | `26.6.0-cuda` | ready after about 395 s; `/v1/responses` returned an image in 1 s at 512x512 and 4 s at 1024x1024 (the default), but every image was colored noise, also with `--quantization-encoding bfloat16`; image quality is unverified on this GPU |
| `MiniMaxAI/MiniMax-Music3` | `26.6.0-cuda` | `/v1/audio/speech` returned a 10 s stereo 44.1 kHz WAV with audible music; the first request took 881 s and peaked near 18.7 GB VRAM |

The negative precision case is an intentional compatibility control: it confirms that the flags select existing encodings rather than converting weights. It does not imply that every FP8 model is unsupported. Use the exact architecture/encoding combinations and example repositories in `max list` or the supported-model matrix.

The same catalog card was then loaded into GPUStack 2.2.3 with one worker on that GPU. Each model was deployed through GPUStack and called through its OpenAI-compatible gateway:

| Model and source | Version and parameters | Result |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B`, Hugging Face | `26.6.0-cuda`, `--device-memory-utilization 0.5 --enable-structured-output --max-length 8192` | running after 283 s including the download (121 s on redeploy); chat and streaming passed; the tool call stayed as `<tool_call>` text in `content`; structured output returned the schema's keys with meaningless values |
| `Qwen/Qwen3-Embedding-0.6B`, local path | `26.6.0-cuda`, `--task embeddings_generation --device-memory-utilization 0.3` | running after 101 s; 1024-dimensional embeddings |
| `sentence-transformers/all-MiniLM-L6-v2`, ModelScope | `26.6.0-cpu` | running after 71 s including the download; 384-dimensional embeddings |

## Parameters

GPUStack accepts additional MAX CLI options in **Backend Parameters**, and the common ones appear as suggestions in the UI. Examples:

```text
--task text_generation
--device-memory-utilization 0.8
--quantization-encoding q4_k
--kv-cache-format bfloat16
--max-length 8192
--reasoning-parser qwen3_5
```

Use `--pretty-print-config` while diagnosing a deployment to record the resolved model, task, devices, and encoding. See the [`max serve` reference](https://docs.modular.com/cli/serve/) for the complete and version-specific option list.

## Licensing

The Modular source repository is Apache-2.0 with LLVM exceptions, while downloading or using the MAX and Mojo SDK or container is governed by the [Modular Community License](https://www.modular.com/legal/community). This backend definition references Modular's official container images; it does not redistribute the MAX SDK.

The Community License is free but is not an open-source software license. Non-production use has no device limit. For commercial production, it permits unlimited x86, ARM, and NVIDIA PTX devices, but limits other accelerator types to eight devices in aggregate unless Modular grants a larger license. An entity starting a commercial production deployment must notify Modular within 30 days, permit identification as an SDK user, and display "Powered by Modular." The license also restricts redistribution, sublicensing, reverse engineering, removal of notices, and unsupported hardware use, and permits collection of usage telemetry.

The included MAX logo is an approved asset from Modular's [trademark policy](https://www.modular.com/legal/trademark). MAX® and Mojo® are trademarks of Modular, Inc. used under license. This community integration is not operated by or endorsed by Modular.

Model checkpoints retain their own licenses. Review and accept all applicable terms before deployment.
