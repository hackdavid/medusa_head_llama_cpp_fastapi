
# Vicuna Medusa Inference Experiment 🚀

## Introduction
This project is an experimental setup designed to test the **Vicuna model** (a LLaMA-based model) using:
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** for efficient CPU/GPU inference
- **Medusa Head** for speculative decoding
- **FastAPI** to serve inference requests with **dynamic batching**.

**Why this experiment?**  
Efficient inference at scale is challenging. Standard token-by-token generation becomes a bottleneck for response time. This project explores how combining **speculative decoding (Medusa)** and **dynamic batching** can drastically improve throughput while maintaining low latency.


Note :  it might not work for now because there is different implementation of forward method in llama_cp to adjust with custom medusa head weights and need a lots of time to analysis and deep-down to find out the correct way.
---

## Model Optimization Methods Mostly Used
### 1. TensorRT-LLM
- Optimized for high-speed GPU inference.
- Requires NVIDIA GPUs and TensorRT installation.
- Complex setup but gives great performance on compatible hardware.

### 2. llama.cpp
- Lightweight, portable, C++ implementation for LLaMA models.
- Supports quantized models (`gguf` format) for **running even on CPUs** efficiently.
- **Chosen for this experiment** due to its simplicity, flexibility, and wide hardware support.

---

## What is Medusa? 
Medusa is a speculative decoding method:
- **Works by predicting multiple future tokens at once** and validating them quickly.
- Instead of waiting for each token's forward pass, Medusa guesses several tokens, validating them with the base model.
- If the guess is correct, it skips computation → **Faster generation!**

> We use **pre-trained Medusa heads** for Vicuna to achieve accelerated decoding without sacrificing much output quality.

---

## What is Dynamic Batching? 📦
Dynamic batching groups multiple inference requests together:
- Instead of processing each request individually (slow), it waits briefly and **combines** incoming requests.
- Processes them together in a single forward pass → **Better hardware utilization and throughput**.

Example:

| Time | Request | Batching |
|-----|--------|---------|
| t1  | User A |  |
| t2  | User B | → Batch A+B |
| t3  | User C |  |
| t4  | User D | → Batch C+D |

This is handled automatically inside the **`batching.py`** file.

---

## Why FastAPI? ⚡
- **Asynchronous**: Handles concurrent requests natively.
- **Fast**: One of the fastest Python web frameworks.
- **Easy integration**: Perfect for exposing model inference APIs.
- **Production-ready**: With tools like **Uvicorn**, you can scale easily.

---

# Project Structure
```bash

├── main.py          # FastAPI server and config loader
├── schema.py        # Request/Response schema definitions
├── model_loader.py      # Loading gguf model and Medusa head
├── model.py             # Token generation and inference logic
├── batching.py          # Dynamic batching logic
├── medusa_with_base.ipynb # Notebook guide for setup
├── requirements.txt     # Required libraries
```

---

# Installation & Setup

## 1. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/vicuna-medusa-experiment.git
cd vicuna-medusa-experiment

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Pre-Setup (Download Necessary Files)

Before running the code, **you must prepare two files**:

### a. Convert HuggingFace Model to GGUF Format
1. **Download Vicuna model locally**:
```python
from huggingface_hub import snapshot_download

model_id = "lmsys/vicuna-7b-v1.3"
snapshot_download(repo_id=model_id, local_dir="vicuna-hf",
                  local_dir_use_symlinks=False, revision="main")
```

2. **Clone llama.cpp**:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

3. **Install llama.cpp dependencies**:
```bash
python -m pip install -r llama.cpp/requirements.txt
```

4. **Convert model to `gguf` format**:
```bash
python llama.cpp/convert_hf_to_gguf.py vicuna-hf \
  --outfile vicuna-7b-v1.gguf \
  --outtype q8_0
```

### b. Download Medusa Head Weights
```python
from huggingface_hub import hf_hub_download

model_id = "FasterDecoding/medusa-vicuna-7b-v1.3"
filename = "medusa_lm_head.pt"

file_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    revision="main",
)
```

> Now you should have:
> - `vicuna-7b-v1.gguf`
> - `medusa_lm_head.pt`

✅ You're ready to move to the coding part!

---

# Coding Walkthrough

### 1. Loading Medusa Model
- `model_loader.py` handles the loading.
- We load Vicuna's Medusa head architecture exactly as described in the [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) repository to maintain correct behavior.

### 2. llama_cpp_python
- This Python binding allows interaction with `gguf` models.
- We pass the Medusa model as a **draft model** argument for speculative decoding.
- **Note:** llama_cpp_python **does not yet** support returning hidden states, so we can't fully customize speculative decoding ourselves.

---

# How to Run 🚀
Start the server using:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Access your API at `http://localhost:8000/docs` to test endpoints interactively!

---

# Important Notes
- **Weights not included**: Due to size constraints, you must manually download and convert weights (follow setup instructions).
- **Current Focus**: The main focus is on building the inference **workflow**, not the response **quality** yet.
- **Future Plans**:
  - Add **benchmarking scripts** (throughput, latency measurement)
  - Investigate improving Medusa quality
  - Implement **custom Medusa training** scripts for fine-tuning

---

# Credits
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
- [Medusa Speculative Decoding](https://github.com/FasterDecoding/Medusa)
- HuggingFace Hub

---
# License
This repository is for **research and experimentation purposes**. Please refer to individual model licenses (Vicuna, Medusa, etc.) for proper usage terms.

---