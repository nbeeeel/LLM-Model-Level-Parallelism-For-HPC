# LLM-Model-Level-Parallelism-For-HPC
4-node model parallel inference for LLaMA 3.2-3B using PyTorch. Splits layers across nodes: Node 0 (embeddings + first layers), Nodes 1-2 (middle layers), Node 3 (final layers + output). Supports distributed computing with Gloo backend, tested on SLURM cluster with GPUs. Includes logging, memory tracking, and multi-prompt testing.
