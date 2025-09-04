# 4-Node Model Parallel Inference for LLaMA 3.2-3B

![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-orange) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.2+-green) ![SLURM](https://img.shields.io/badge/SLURM-Cluster-red)

This repository implements a **4-node model parallel inference pipeline** for the LLaMA 3.2-3B model using PyTorch's distributed computing capabilities with the Gloo backend. The model is split across four nodes, with each handling a portion of the transformer layers, optimized for GPU-accelerated clusters managed by SLURM.

---

## 📖 Overview

The code distributes the LLaMA 3.2-3B model across four nodes for efficient inference:
- **Node 0**: Handles input embeddings and the first set of transformer layers.
- **Nodes 1-2**: Process middle transformer layers.
- **Node 3**: Manages final transformer layers, normalization, and output head.

The implementation includes:
- **Dynamic layer splitting** based on the number of nodes.
- **Distributed communication** using PyTorch's Gloo backend.
- **Memory usage tracking** for GPUs.
- **SLURM integration** for cluster deployment.
- **Test suite** with multiple prompts to evaluate performance.

This setup is ideal for researchers and engineers working on large-scale model inference in distributed environments.

---

## 🚀 Features

- **Model Parallelism**: Splits LLaMA 3.2-3B layers across 4 nodes for efficient computation.
- **Gloo Backend**: Reliable distributed communication with configurable timeouts.
- **SLURM Support**: Seamlessly integrates with SLURM for cluster management.
- **Memory Monitoring**: Tracks GPU memory usage before and after model loading.
- **Flexible Configuration**: Adjustable parameters for max tokens, temperature, and model path.
- **Test Prompts**: Evaluates performance with a set of predefined prompts.

---

## 📋 Requirements

- **Hardware**: 4 nodes with at least 1 GPU per node (CUDA 12.2 compatible).
- **Memory**: Minimum 48GB RAM per node.
- **Software**:
  - Python 3.9+
  - PyTorch 2.0+ with CUDA 12.2
  - Transformers library (`huggingface_hub`)
  - SLURM cluster environment
- **Model**: LLaMA 3.2-3B weights (stored at `/mnt/lustre/user/llama3.2-3b-instruct`).

---

## 🛠 Installation

1. **Set up the environment**:
   ```bash
   module load python/3.9 cuda/12.2
   source /mnt/lustre/user/llm-env/bin/activate
   pip install torch transformers

Clone the repository:
bashgit clone https://github.com/your-username/4node-llama-inference.git
cd 4node-llama-inference

Prepare the model:
Ensure the LLaMA 3.2-3B model is available at /mnt/lustre/user/llama3.2-3b-instruct. Update MODEL_PATH in model_parallel_4node.py if necessary.
Configure SLURM:
Modify Inference.slurm to match your cluster's node names, GPU count, and resource requirements.

#### ▶️ Usage

### Run the inference:
Submit the SLURM job to execute the 4-node inference:
bashsbatch Inference.slurm

### Output:

Logs are saved to 4node_test_%j.out and 4node_test_%j.err (where %j is the SLURM job ID).
Console output includes prompt responses, token generation stats, and performance metrics (tokens/sec).

### Test Prompts:
The script tests five prompts:

- What is artificial intelligence?
- How do neural networks work?
- Explain machine learning simply.
- What is distributed computing?
- How do transformers work?

Results include response text, token count, generation time, and throughput.

#### 📊 Performance
The script measures:

- Total tokens generated across all prompts.
- Average token generation speed (tokens/second).
- Total runtime for all prompts.
- Memory usage per node (before and after model loading).

### Example output:
text=======================================
4-NODE LAYER SPLITTING TEST
=======================================
Model: LLaMA 3.2-3B
Layer distribution:
  Node 0: layers 0-7 (8 layers) + embeddings
  Node 1: layers 8-15 (8 layers)
  Node 2: layers 16-23 (8 layers)
  Node 3: layers 24-31 (8 layers) + norm + lm_head
=======================================
Prompt: What is artificial intelligence?
Response: Artificial intelligence is the simulation of human intelligence in machines, enabling them to perform tasks like reasoning, learning, and decision-making.
Stats: 20 tokens, 15.5 tok/s
...
RESULTS SUMMARY
=======================================
Prompts: 5
Total tokens: 120
Avg speed: 14.8 tok/s
Total time: 8.12s
=======================================

### ⚙️ Code Structure

model_parallel_4node.py:

- Initializes distributed environment with Gloo backend.
- Implements FourNodeModel class for layer splitting and model loading.
- Handles inference with token generation and distributed communication.
- Logs memory usage and performance metrics.

Inference.slurm:

- Configures SLURM job for 4 nodes, 1 GPU per node, and 48GB RAM.
- Sets up environment with Python 3.9, CUDA 12.2, and virtual environment.
- Runs the inference script with srun.

### 🔧 Configuration
Key parameters in model_parallel_4node.py:

- MODEL_PATH: Path to LLaMA 3.2-3B model weights.
- MASTER_PORT: Port for distributed communication (default: 12355).
- MAX_TOKENS: Maximum tokens to generate per prompt (default: 100).
- TEMPERATURE: Sampling temperature for token generation (default: 0.8).

Modify these in the script to suit your needs.

### 🐛 Troubleshooting

Distributed setup failure:

- Check MASTER_ADDR and MASTER_PORT in logs.
- Ensure network interfaces are correctly detected (detect_network_interface).
- Verify Gloo backend settings (GLOO_SOCKET_IFNAME, GLOO_TIMEOUT_SECONDS).

#### Memory errors:

- Monitor GPU memory usage in logs.
- Reduce batch size or adjust layer distribution if needed.


#### SLURM issues:

- Confirm node availability and resource allocation in Inference.slurm.
- Check SLURM logs (4node_test_%j.err) for errors.

#### 📜 License
This project is licensed under the MIT License. See LICENSE for details.

#### 🙌 Contributing
Contributions are welcome! Please:

- Fork the repository.
- Create a feature branch (git checkout -b feature/your-feature).
- Commit changes (git commit -m 'Add your feature').
- Push to the branch (git push origin feature/your-feature).
- Open a pull PR.


📬 Contact
