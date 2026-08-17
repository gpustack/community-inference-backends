# GPUStack Community Inference Backends

This repository hosts **community-maintained inference backends** for GPUStack.

It is designed to provide a clear, lightweight, and open way for inference engine developers and advanced users to integrate their backends with GPUStack, without requiring changes to GPUStack core.

If you have an inference engine that already works in Docker and serves models reliably, you are welcome here.

---

## Background & Goals

The inference ecosystem evolves quickly across models, hardware platforms, and runtimes. No single project can maintain first-class support for every inference engine.

GPUStack positions itself as a high-performance and flexible **Model-as-a-Service (MaaS)** platform. This repository exists to:

- Embrace diversity in inference engines
- Lower the integration barrier for new backends
- Clearly separate **core responsibilities** from **community ownership**
- Allow proven community backends to grow naturally through real usage

---

## What Is a Community Inference Backend

A Community Inference Backend is:

- Integrated with GPUStack through a standardized spec
- Maintained by the community or upstream project
- Explicitly labeled as **Community**, not Built-in
- Free to evolve independently of GPUStack core releases

GPUStack **does not guarantee**:
- Runtime correctness
- Model compatibility
- Inference quality or performance

---

## Available Community Backends

The following community backends are currently available:

### Backend Categories

- **LLM**
  - [llama.cpp](https://github.com/ggml-org/llama.cpp)
  - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) [Docs](https://nvidia.github.io/TensorRT-LLM/)
  - [TokenSpeed](https://github.com/lightseekorg/tokenspeed) [Docs](https://lightseek.org/tokenspeed/)

- **Vision / OCR**
  - [MinerU](https://github.com/opendatalab/MinerU) [Docs](https://opendatalab.github.io/MinerU/)
  - [PaddleX-GenAI-Server](https://github.com/PaddlePaddle/PaddleX) [Docs](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html)

- **Embedding / Reranker / Sequence Classification**
  - [Text-Embeddings-Inference](https://github.com/huggingface/text-embeddings-inference) [Docs](https://huggingface.co/docs/text-embeddings-inference/index)
  - [mis-tei](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)

- **Audio**
  - Text-to-Speech
    - [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)

---

## Repository Structure

Each backend lives in its own directory and should include:

- `spec.yaml` Backend metadata and definition (name, versions, parameters, etc.)
  Refer to [`spec-template.yaml`](./spec-template.yaml) for an example.

- `README.md`
  Usage instructions, supported models, configuration, and limitations

- `logo.png` 
  Backend logo for display in the GPUStack inference backends catalog

Additional files may be included if required by the backend.

## Who Should Contribute

You are encouraged to contribute if you are:

- An inference engine developer or advanced user
- Running a backend successfully in real environments
- Experimenting with new hardware, accelerators, or runtimes
- Interested in sharing a useful backend with the community

---

## Minimal Requirements

Before submitting a backend, make sure:

- It can run in a container and serve APIs
- You know how to start it with a command
- It exposes a health or API endpoint GPUStack can check
- You understand its model compatibility at a basic level

That’s enough to get started.

---

## How to Contribute

1. Fork this repository
2. Create a new directory for your backend
3. Add `spec.yaml`, `README.md` and `logo.png` (Refer to [`spec-template.yaml`](./spec-template.yaml) for an example)
4. Open a Pull Request

In your PR description, briefly explain:
- What problem this backend solves
- Who maintains it
- Whether it is used in real environments

---

## Maintenance & Ownership

By contributing a backend, you agree to:

- Own its long-term maintenance
- Fix obvious breakages when possible
- Update documentation if behavior changes

Backends that become unmaintained may be marked accordingly or removed.

---

## Built-in vs Community Backends

- **Built-in backends** are maintained by GPUStack core
- **Community backends** live in this repository and are community-owned

Outstanding community backends — proven through real usage, adoption, and close integration — *may* be promoted to Built-in backends in the future.

---

## Why This Matters

This repository helps GPUStack remain:

- Neutral and extensible
- Open to innovation
- Scalable without overloading core maintainers

If you build a useful inference backend, this is where it can naturally connect with GPUStack users and grow.

We look forward to your contributions.
