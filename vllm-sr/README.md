# vllm-sr (vLLM Semantic Router model runtime)

This backend runs the [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) **model runtime** (`vllm-srun`) as a GPUStack custom backend, serving **decision models** over the System One decision API.

A decision model reads an input (text or JSON, optionally with images and videos) once and answers typed questions about it in a single call — pick one of several options, say yes or no, rate on a scale — returning a probability for every answer, without generating text.

## Supported models

The runtime detects the model family from the package. Deploy the model with its Hugging Face repository id, e.g. `vllm-sr/d3-mini`.

| Family | Models | Question types |
| --- | --- | --- |
| Decision 3.0 | `d3-ultra`, `d3-turbo`, `d3`, `d3-flash`, `d3-mini`, `d3-nano`, `d3-lite`, `d3-edge` | choice, yes/no (noul), score; text, image and video inputs (`d3-turbo` and `d3-ultra` also answer box/track) |
| Decision 2.0 | `Decision-2.0-Nox-4B` | choice, yes/no, score |
| Decision 1.0 | `Decision-1.0-Eos-0.8B` | choice, yes/no, score |
| Vela 2.0 | `Vela-2.0-*` | choice, yes/no, score, set, span |

Sizing guide for the Decision 3.0 family (BF16 weights): `d3-ultra`/`d3-turbo`/`d3` (26B, ~49 GB) need a GPU with 64 GB+ of memory; `d3-flash` (8.4B, ~17 GB) needs 24 GB+; `d3-mini` (4.5B, ~9 GB) fits 16 GB; `d3-nano` (2.2B) and smaller also run on CPU.

## How GPUStack runs it

The published images start the Semantic Router by default; this backend overrides the entrypoint with `vllm-srun` and starts a single model, pinning the device per platform (`cuda:0`, `rocm:0`, `cpu`):

```
vllm-srun serve <model_path> --device cuda:0 --host <worker_ip> --port <port> --served-model-name <model_name>
```

- **API**: `POST /v1/systemone` (System One decision API), `GET /v1/models`, `GET /health`
- **Health check**: `/health`
- GPUStack marks these deployments as decision-category models and skips the OpenAI chat/embedding probe against them.
- **One replica uses one GPU**: the runtime has no tensor parallelism; scale out with replicas. GPUStack injects only the GPUs allocated to an instance, renumbered from 0, so `cuda:0` is always the first allocated GPU regardless of its physical index.
- The device is pinned instead of `auto` on purpose: on an out-of-memory GPU, `auto` silently falls back to the CPU, where the load-time golden verification fails on hardware without AVX512-BF16. Pinned, the same situation fails fast with a clear out-of-memory error.

### Example request

```bash
curl -s http://<worker_ip>:<port>/v1/systemone -H 'content-type: application/json' -d '{
  "model": "d3-mini",
  "state": "Ticket: the invoice total is wrong after the plan change.",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team handles this?",
              "criteria": {"billing": "Billing", "tech": "Technical support"}},
    "urgent": {"type": "noul", "instructions": "Is the customer asking for an urgent fix?"}
  }
}'
```

The reply carries the picked option with a probability per option:

```json
{
  "model": "d3-mini",
  "answers": {
    "route": {"type": "choice", "choice": "billing",
              "probabilities": {"billing": 0.9968, "tech": 0.0032}, "confidence": 0.9936},
    "urgent": {"type": "noul", "noul": 0.3714}
  },
  "usage": {"input_tokens": 207, "output_tokens": 0}
}
```

## Versions

| Version | Image | Hardware |
| --- | --- | --- |
| cuda | `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` | NVIDIA GPUs |
| rocm | `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest` | AMD GPUs |
| cpu | `ghcr.io/vllm-project/semantic-router/vllm-sr:latest` | CPU (small models) |

The images track upstream `main` (`latest` and per-commit tags are published on GHCR). `d3-ultra` support requires an image published after 2026-10-25; current `latest` images include it.

## Notes and limitations

- `vllm-srun` validates the request's `model` field against `--served-model-name`: requests carrying any other name are rejected with `model_not_found`.
- The runtime auto-calibrates batching (`max_num_batched_tokens`, `max_num_seqs`) per device at startup; override via backend parameters if needed.
- First start of a Hub model downloads the weights inside the container; use the `--offline` and `--cache-dir` backend parameters (and GPUStack cache settings) to control that.
- Upstream validates the CUDA images on NVIDIA GPUs less thoroughly than ROCm (the project's reference hardware is AMD Instinct).
