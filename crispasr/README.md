# CrispASR

[CrispASR](https://github.com/CrispStrobe/CrispASR) is a unified C++ speech engine and whisper.cpp fork that runs 145+ ASR and TTS backends from a single binary with GGUF models, without Python dependencies. In server mode it exposes an OpenAI-compatible audio transcription API plus TTS, diarization, VAD, punctuation restoration, and streaming.

- Backend name: `CrispASR`
- Images: `ghcr.io/crispstrobe/crispasr:main-cuda` (CUDA 13) / `ghcr.io/crispstrobe/crispasr:main` (CPU)
- Health endpoint: `/health`
- Model discovery: `/v1/models`
- Backend definition: [spec.yaml](./spec.yaml)
- Upstream server documentation: [docs/server.md](https://github.com/CrispStrobe/CrispASR/blob/main/docs/server.md)

## Hardware and model requirements

- **`main-cuda`**: requires an NVIDIA driver R535+ (CUDA 13 runtime). Supports compute capabilities sm_75 through sm_120.
- **`main`** (CPU): requires AVX2 support (Intel Haswell or newer, AMD Zen+).

The model is a single GGUF file. Multi-file backends (e.g. moss-tts) require additional companion parameters such as `--codec-model`; these are not covered by the single `{{model_path}}` substitution and must be passed via `common_parameters`.

## Deploy with GPUStack

1. Enable **CrispASR** from GPUStack's community inference backends catalog.
2. Create a model deployment with a GGUF speech model and select the desired backend version (`cuda` or `cpu`).
3. Select the GPUs needed by the model on a worker. GPUStack supplies the model path, HTTP port, and model name.
4. Use `common_parameters` for additional flags such as `--backend`, `-l` (language), `--n-gpu-layers`, `--threads`, `--server-workers`, `--punc-model`, `--vad`, or `--diarize`.

The inference backend is auto-detected from the GGUF model metadata. Override with `--backend` if auto-detection picks the wrong one.

## Verify the deployment

Use the deployment's endpoint in these examples. When accessing the GPUStack gateway, include its required API-key authorization header.

```bash
curl --fail http://localhost:8080/health
curl --fail http://localhost:8080/v1/models

curl --fail http://localhost:8080/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

The health endpoint returns HTTP 503 while the model is loading and HTTP 200 once the model is resident.

## Limitations

- `main-cuda` and `main` are floating image tags; this definition does not pin an image digest.
- TTS voice cloning requires consent attestation per the upstream [EU AI Act notes](https://github.com/CrispStrobe/CrispASR/blob/main/docs/eu-ai-act.md).
- GPUStack does not guarantee inference quality, model compatibility, or runtime correctness for community backends.
- Multi-file model setups (companion GGUF files) are not covered by the single `{{model_path}}` substitution.

## Source and maintenance

The serving implementation and container images are maintained in [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR). Report serving issues there and community integration issues in this repository.
