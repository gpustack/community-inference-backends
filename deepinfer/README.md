# DeepInfer

[DeepInfer Server](https://github.com/gpustack/DeepInfer-server) provides
OpenAI-compatible text chat completions for DeepSeek-V4.1-Flash using DeepInfer
and deepseek-recipe. It supports JSON and SSE responses, reasoning content, and
automatic tool-call parsing.

- Backend name: `DeepInfer`
- Image: `quay.io/gpustack/deepinfer-server:latest`
- Health endpoint: `/health`
- Model discovery: `/v1/models`
- Backend definition: [spec.yaml](./spec.yaml)

## Hardware and model requirements

**This backend only supports NVIDIA B200/GB200 (SM100) GPUs on Linux.** Other GPU
models and multi-worker deployments are not supported.

Deploy a complete `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint, including its
configuration, tokenizer, and safetensors weights. The worker needs sufficient
GPU memory for the model and KV cache, sufficient host memory for the
CPU-resident Engram tables, an NVIDIA driver compatible with the image's CUDA
13.2 runtime, and NVIDIA Container Toolkit.

Working RDMA and interconnects meeting DeepInfer/DeepEP requirements are required.
Expose the worker's RDMA devices (`/dev/infiniband`) to the inference container
and configure the host fabric before deployment. The image does not configure
the host network or device passthrough.

## Deploy with GPUStack

1. Enable **DeepInfer** from GPUStack's community inference backends catalog.
2. Create a model deployment with the complete DeepSeek-V4.1-Flash checkpoint
   and select the `latest` backend version.
3. Select all GPUs needed by the model on a single worker. GPUStack supplies the
   model path, HTTP port, model name, and selected GPU count.
4. Configure the model environment for the worker's RDMA fabric and ensure the
   JIT cache directory is writable and persistent.

The default command uses expert parallelism (EP) equal to the selected GPU count
and data parallelism (DP) of 1. Context parallelism (CP) is derived automatically;
`DP * CP = EP`. For the upstream eight-B200 example, select all eight GPUs on the
same worker, giving EP=8, DP=1, and CP=8. This example is not a verified minimum
GPU count or memory requirement. If overriding parallelism, preserve this
relationship.

The defaults allow a model length of 32768, 16 concurrent sequences, 32768 batched
tokens, and a startup timeout of 3600 seconds for cold weight loading and JIT
compilation. Adjust these parameters for available memory and workload. The
HTTP server starts listening only after model initialization completes.

### IPC and memory limits

Keep `GPUSTACK_HOST_IPC=true` in the model environment. The entrypoint sets
unlimited locked memory and a 64 MiB stack limit before starting the server.
The container runtime must allow these limits; a failed `ulimit` prevents startup.

### RDMA configuration

RDMA settings are fabric-specific and are intentionally omitted from the backend
defaults. The upstream eight-B200 validation worker used these model environment
values:

```yaml
NCCL_IB_HCA: "=mlx5_0"
NCCL_IB_SL: "0"
EP_OVERRIDE_RDMA_SL: "0"
```

The leading `=` selects the exact HCA name. Use the HCA and service levels
appropriate for your own fabric. These values are an example, not universal
requirements for B200/GB200 systems.

If Engram initialization reports `GIN: DevComm setup failed on all available
backends`, check RDMA device access, HCA selection, and NCCL fabric configuration.
If setting `MASTER_PORT`, ensure `MASTER_PORT - 50` is free and different from
the HTTP port; DeepInfer uses it for distributed initialization.

### Persistent JIT cache

The backend sets
`DEEPINFER_JIT_CACHE_DIR=/var/lib/gpustack/cache/deepinfer/jit`.
Provide a writable persistent mount covering this path. GPUStack mirrored
deployments can inherit the worker's cache mount; otherwise configure a mount
explicitly. Override the environment variable if the worker uses a different
data directory. This cache persists compiled kernels, not GPU KV cache, and
does not eliminate model loading on restart. GPUStack supplies downloaded
checkpoints through `{{model_path}}` from its model cache.

## Verify the deployment

Use the deployment's endpoint and served model name in these examples. When
accessing the GPUStack gateway, include its required API-key authorization header.

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models

curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "thinking": {"type": "disabled"},
    "max_tokens": 128,
    "stream": false
  }'
```

To check streaming, set `"stream": true` and add `-N` to curl. Thinking mode
follows recipe defaults, and `max_tokens` includes reasoning tokens; the example
disables thinking for an initial deployment check. The upstream repository also
provides `python scripts/smoke.py --base-url http://localhost:8000`.

## Limitations and validation

- Text inputs and `n=1` only. Images, logprobs, forced tool selection, and
  structured output constraints are not supported.
- Tool calls can be automatic or disabled. Clients execute tools; the service
  generates and parses tool calls.
- Responses/Messages APIs, cross-request prefix caching, and disaggregated
  prefill/decode deployment are not provided.
- Speculative decoding is disabled by default. See the upstream documentation
  for the relevant DeepInfer environment variables.
- Requests exceeding capacity return HTTP 429. A worker failure makes the
  health endpoint return HTTP 503 and requires restarting the service.
- `latest` is a mutable image tag; this definition does not pin an image digest.

The [upstream deployment notes](https://github.com/gpustack/DeepInfer-server#gpustack-integration)
report successful DeepEP Engram and non-streaming/streaming Chat Completions
checks on an eight-B200 worker. This contribution does not add a new GPU runtime
validation or claim a GB200 validation result. Validate the image, model loading,
and inference on your target hardware.

## Source and maintenance

The serving implementation and container source are maintained in
[gpustack/DeepInfer-server](https://github.com/gpustack/DeepInfer-server). Report
serving issues there and community integration issues in this repository.

The catalog logo is the [DeepSeek organization avatar](https://github.com/deepseek-ai),
reused from the upstream backend definition. It is not a separate DeepInfer logo.
