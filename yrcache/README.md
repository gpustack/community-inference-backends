# YRCache

This community inference backend enables GPUStack to equip **vLLM** and **SGLang** inference engines with **YRCache** distributed KV Cache sharing capabilities, eliminating redundant computation on identical prompt prefixes and significantly improving inference throughput in multi-turn conversation and long-context scenarios.

---

## 1. YRCache Architecture

YRCache is deployed as a plugin on each GPU inference node, running in-process with the inference engine (vLLM / SGLang). It is logically split into two layers: the **Connector layer** (Python) adapts to the engine's KV transfer interface, while the **Cache Engine layer** (C++) handles high-performance concurrent I/O across three storage tiers — CPU Memory, NVMe SSD, and an optional shared filesystem. A Redis instance manages distributed metadata for cross-node cache lookups. At runtime, the inference engine's Scheduler process queries YRCache to check which KV Cache blocks already exist, and Worker processes load cached blocks into GPU memory on hits and write newly computed blocks back for reuse. All of this requires no standalone service.

---

## 2. What Problem It Solves

In standard LLM inference, KV Cache for identical prefixes (system prompts, multi-turn conversation history) is recomputed on every request. YRCache caches these blocks so subsequent requests can reuse them directly, eliminating redundant computation overhead.

---

## 3. Key Features

- **Three-Tier Storage Architecture** — CPU Memory / NVMe SSD (multi-device concurrency supported) / Shared Filesystem. Tiers can be enabled independently or combined as needed.
- **Cross-Node KV Cache Sharing** — Shared filesystem tier uses Redis for metadata management and ZMQ for cross-node cache queries.
- **Seamless vLLM & SGLang Integration** — Runs as a Connector plugin within the same process. No standalone service required.

---

## 4. Prerequisites

Supports GPUStack ≥ 2.1. Compatible with NVIDIA GPUs (CUDA driver provided by the host; the inference engine base image bundles its own CUDA toolkit). See [YRCache-Compatibility](https://qlxi12kq64.feishu.cn/wiki/Idj2wBqI5iD5v9kZaqjcEME0nkU?from=from_copylink).

---

## 5. Supported Models

Covers mainstream model families including DeepSeek, Qwen, and more. Models can be sourced from Hugging Face, ModelScope, or local paths. See [YRCache-Compatibility](https://qlxi12kq64.feishu.cn/wiki/Idj2wBqI5iD5v9kZaqjcEME0nkU?from=from_copylink).

---

## 6. Configuration

`yrcache.yaml` is the configuration file for the YRCache distributed KV cache system. It configures cache policies, storage paths, network communication, and other parameters.

At least one of the three cache tiers (`cpu_cache_config` / `local_disk_cache_config` / `shared_storage_cache_config`) must be enabled.

Write `yrcache.yaml` and place it on each Worker node. For the yrcache.yaml reference, see [yrcache.yaml](https://qlxi12kq64.feishu.cn/wiki/Ev3zwEojIiT80wkoC89cDtlOn9f?from=from_copylink).

---

## 7. Exporter Dashboard

YRCache includes a built-in Prometheus Exporter that can be paired with Grafana dashboards to monitor key metrics such as cache hit rate, request count, and cumulative KV Cache volume, helping users track cache effectiveness and inference performance in real time. See [Metrics Monitoring Deployment](https://qlxi12kq64.feishu.cn/wiki/SFvDwdSOIiBssakc0k9cZ8xJnLc?from=from_copylink).

---

## 8. GPUStack UI Deployment

### 8.1 Deploying the GPUStack Image

```bash
# Pull the GPUStack image
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/gpustack/gpustack:v2.2.1

docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/gpustack/gpustack:v2.2.1 gpustack/gpustack:v2.2.1

# Run the GPUStack image
docker run -d --name gpustack --restart=unless-stopped --gpus all --network=host --ipc=host -v gpustack-data:/var/lib/gpustack gpustack/gpustack:v2.2.1 --port 9091

# Reset the admin password
docker exec gpustack gpustack reset-admin-password --server-url http://localhost:9091
```

Once deployed, the Web UI can be accessed at `http://<host-ip>:9091`.

![](images/1.png)

### 8.2 Adding Worker Nodes

#### 8.2.1 Add Node

![Image](images/2.png)

![Image](images/3.png)

![Image](images/4.png)

![Image](images/5.png)

![Image](images/6.png)

#### 8.2.2 Run gpustack-worker

Run the command pasted from the above UI on each node you wish to add. **Note: If the L3 shared storage tier is enabled, add the shared storage path specified in the YRCache configuration file.**

```bash
# Cache volume
mkdir -p /mnt/nvme1n1p3/gpustack/cache && chmod 777 /mnt/nvme1n1p3/gpustack/cache
# Extra volume — this directory contains the yrcache.yaml file
mkdir -p /workspaces && chmod 777 /workspaces
# Shared storage path specified in the YRCache config
mkdir -p /nvme0n1p10/yrcache_vllm && chmod 777 /nvme0n1p10/yrcache_vllm
# YRCache local disk cache directories
mkdir -p /nvme1n1p2/yrcache_vllm && chmod 777 /nvme1n1p2/yrcache_vllm
```

```bash
sudo docker run -d --name gpustack-worker \
      -e "GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME=gpustack-worker" \
      -e "GPUSTACK_TOKEN=gpustack_****" \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --volume /workspaces:/workspaces \
      --volume /nvme0n1p10:/nvme0n1p10 \
      --volume /nvme1n1p2:/nvme1n1p2 \
      --volume /mnt/nvme1n1p3/gpustack/cache:/var/lib/gpustack/cache \
      --runtime nvidia \
      quay.io/gpustack/gpustack:v2.2.1 \
      --server-url http://192.168.255.81:9091 \
      --worker-ip 192.168.255.81
```

#### 8.2.3 Verify with docker ps

![Image](images/19.png)

#### 8.2.4 Verify in the UI

![Image](images/7.png)

Node addition is now complete.

### 8.3 Adding an Inference Backend

#### 8.3.1 Community

![Image](images/8.png)

![Image](images/9.png)

#### 8.3.2 Custom

**vLLM/SGLang-YRCache Image Download** 

> vllm:yanrong-registry.cn-hangzhou.cr.aliyuncs.com/yrcache_namespace/yrcache-gpustack:vllm-0.24.0-yrcache-1.7.2
>
> sglang:yanrong-registry.cn-hangzhou.cr.aliyuncs.com/yrcache_namespace/yrcache-gpustack:sglang-0.5.15-yrcache-1.7.2
>

**Add via YAML Mode**

Select YAML mode. Load `spec.yaml` into GPUStack. For the spec.yaml reference, see [spec.yaml](https://qlxi12kq64.feishu.cn/wiki/SGOfwwK2Fibwh2kOytCc53zrnkg?from=from_copylink).

![Image](images/10.png)

![](images/11.png)

1. Write `yrcache.yaml` and place it on each Worker node. For the yrcache.yaml reference, see [yrcache.yaml](https://qlxi12kq64.feishu.cn/wiki/Ev3zwEojIiT80wkoC89cDtlOn9f?from=from_copylink)
2. When deploying the model, select the backend whose `backend_name` matches the one defined in spec.yaml.

### 8.4 Adding Model Files

Go to the UI: Model Service → Model Files → Add Model File. Enter the local model file path, and select the node added above.

![Image](images/12.png)

![Image](images/13.png)

### 8.5 Deploying Models

**Start Deployment**

![Image](images/14.png)

**Select Backend**

For custom backends, select the backend whose `backend_name` matches the one defined in spec.yaml.

![Image](images/15.png)

**Add Parameters — vLLM**

![Image](images/16.png)

**Add Parameters — SGLang**

![](images/17.png)

**Deployment Successful**

![Image](images/18.png)

---

## 9. Maintenance

This backend is maintained by the YRCache team.

Contact:

wangpengfei@yanrongyun.com

limengran@yanrongyun.com
