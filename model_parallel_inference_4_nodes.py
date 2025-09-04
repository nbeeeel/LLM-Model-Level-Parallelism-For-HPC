#!/usr/bin/env python3
"""
4-node layer splitting inference
Node 0: embeddings + first layers
Nodes 1-2: middle layers
Node 3: final layers + output
"""

import os
import torch
import torch.distributed as dist
import time
import socket
import subprocess
from datetime import timedelta
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM

# Config
MODEL_PATH = "/mnt/lustre/user/llama3.2-3b-instruct"
MASTER_PORT = "12355"
MAX_TOKENS = 100
TEMPERATURE = 0.8

def log(message):
    hostname = socket.gethostname()
    rank = int(os.environ.get('SLURM_PROCID', 0))
    print(f"[{hostname}:R{rank}] {message}", flush=True)

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{allocated:.1f}GB/{total:.1f}GB"
    return "No CUDA"

def detect_network_interface():
    hostname = socket.gethostname()

    # Try to find 192.168.20.x interface
    try:
        result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'inet 192.168.20' in line:
                interface = line.strip().split()[-1]
                return interface
    except:
        pass

    # Fallback mapping
    interface_map = {
        'master-node': 'ens1f1',
        'node01': 'ens1f1',
        'node02': 'ens1f0np0',
        'node03': 'ens1f1'
    }

    return interface_map.get(hostname, 'ens1f1')

def setup_distributed():
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    # Set network interface
    network_interface = detect_network_interface()
    os.environ['GLOO_SOCKET_IFNAME'] = network_interface

    # Get master node
    try:
        result = subprocess.run(['scontrol', 'show', 'hostname',
                               os.environ['SLURM_JOB_NODELIST']],
                              capture_output=True, text=True)
        nodes = result.stdout.strip().split('\n')
        master_addr = nodes[0] if nodes else 'master-node'
    except:
        master_addr = 'master-node'

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    try:
        dist.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10)
        )
        log("Distributed setup complete")
        return rank, world_size
    except Exception as e:
        log(f"Distributed setup failed: {e}")
        return None, None

def send_tensor(tensor, dst):
    if tensor.device.type == 'cuda':
        cpu_tensor = tensor.cpu().contiguous()
    else:
        cpu_tensor = tensor.contiguous()
    dist.send(cpu_tensor, dst=dst)

def recv_tensor(shape, dtype, src, device):
    tensor = torch.zeros(shape, dtype=dtype)
    dist.recv(tensor, src=src)
    return tensor.to(device)

class FourNodeModel:
    def __init__(self, model_path, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # Load config and tokenizer
        self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Calculate layer distribution
        total_layers = self.config.num_hidden_layers
        layers_per_node = total_layers // world_size
        remainder = total_layers % world_size

        # Distribute layers
        self.layer_assignments = []
        start = 0
        for i in range(world_size):
            extra = 1 if i < remainder else 0
            node_layers = layers_per_node + extra
            end = start + node_layers
            if i == world_size - 1:
                end = total_layers
            self.layer_assignments.append((start, end))
            start = end

        self.my_start, self.my_end = self.layer_assignments[rank]
        log(f"Assigned layers {self.my_start}-{self.my_end-1} ({self.my_end-self.my_start} layers)")

        log(f"Memory before loading: {get_memory_usage()}")
        self._load_components()
        log(f"Memory after loading: {get_memory_usage()}")

    def _load_components(self):
        # Load full model
        full_model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            local_files_only=True
        )

        if self.rank == 0:
            # First node: embeddings + first layers
            log("Loading embeddings + first layers")
            self.embed_tokens = full_model.model.embed_tokens.to(self.device)
            self.rotary_emb = full_model.model.rotary_emb.to(self.device)

            self.layers = torch.nn.ModuleList()
            for i in range(self.my_start, self.my_end):
                layer = full_model.model.layers[i].to(self.device)
                self.layers.append(layer)

        elif self.rank == self.world_size - 1:
            # Last node: final layers + output
            log("Loading final layers + output")
            self.layers = torch.nn.ModuleList()
            for i in range(self.my_start, self.my_end):
                layer = full_model.model.layers[i].to(self.device)
                self.layers.append(layer)

            self.norm = full_model.model.norm.to(self.device)
            self.lm_head = full_model.lm_head.to(self.device)

        else:
            # Middle nodes: just their layers
            log(f"Loading middle layers {self.my_start}-{self.my_end-1}")
            self.layers = torch.nn.ModuleList()
            for i in range(self.my_start, self.my_end):
                layer = full_model.model.layers[i].to(self.device)
                self.layers.append(layer)

        # Cleanup
        del full_model
        torch.cuda.empty_cache()

def generate_response(prompt, model, rank, world_size, device):
    if rank == 0:
        # Master node processing
        log(f"Processing: '{prompt}'")

        inputs = model.tokenizer(prompt, return_tensors="pt")
        current_ids = inputs['input_ids'].to(device)

        generated_tokens = 0
        start_time = time.time()
        response_text = ""

        for step in range(MAX_TOKENS):
            try:
                with torch.no_grad():
                    batch_size, seq_len = current_ids.shape

                    # Embeddings and setup
                    hidden_states = model.embed_tokens(current_ids)
                    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
                    cache_position = torch.arange(0, seq_len, dtype=torch.long, device=device)
                    cos, sin = model.rotary_emb(hidden_states, position_ids)

                    # Process my layers
                    for layer in model.layers:
                        layer_output = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings=(cos, sin)
                        )
                        hidden_states = layer_output[0]

                    # Send to next node
                    seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64)
                    send_tensor(seq_len_tensor, dst=1)
                    send_tensor(hidden_states, dst=1)
                    send_tensor(position_ids, dst=1)
                    send_tensor(cache_position, dst=1)
                    send_tensor(cos, dst=1)
                    send_tensor(sin, dst=1)

                    # Receive final result
                    logits = recv_tensor(
                        shape=(batch_size, seq_len, model.config.vocab_size),
                        dtype=torch.float16,
                        src=world_size-1,
                        device=device
                    )

                    # Sample next token
                    last_logits = logits[0, -1, :] / TEMPERATURE
                    probs = torch.softmax(last_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                    generated_tokens += 1

                    new_token_text = model.tokenizer.decode(next_token.item())
                    response_text += new_token_text
                    print(new_token_text, end="", flush=True)

                    # Stop conditions
                    if next_token.item() == model.tokenizer.eos_token_id:
                        break
                    if step > 15 and new_token_text.strip().endswith(('.', '!', '?')):
                        break

            except Exception as e:
                log(f"Error in step {step}: {e}")
                break

        # Send stop signal
        stop_signal = torch.tensor([-1], dtype=torch.int64)
        send_tensor(stop_signal, dst=1)

        generation_time = time.time() - start_time

        print(f"\nStats: {generated_tokens} tokens, {generated_tokens/generation_time:.2f} tok/s")

        return {
            'prompt': prompt,
            'response': response_text.strip(),
            'tokens': generated_tokens,
            'time': generation_time,
            'speed': generated_tokens / generation_time if generation_time > 0 else 0
        }

    elif rank == world_size - 1:
        # Last node: final processing
        step_count = 0
        while True:
            try:
                # Receive from previous node
                seq_len_tensor = recv_tensor((1,), torch.int64, src=rank-1, device='cpu')
                seq_len = seq_len_tensor.item()

                if seq_len < 0:  # Stop signal
                    break

                # Receive tensors
                hidden_states = recv_tensor((1, seq_len, model.config.hidden_size), torch.float16, src=rank-1, device=device)
                position_ids = recv_tensor((1, seq_len), torch.int64, src=rank-1, device=device)
                cache_position = recv_tensor((seq_len,), torch.int64, src=rank-1, device=device)
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                cos = recv_tensor((1, seq_len, head_dim), torch.float16, src=rank-1, device=device)
                sin = recv_tensor((1, seq_len, head_dim), torch.float16, src=rank-1, device=device)

                with torch.no_grad():
                    # Process my layers
                    for layer in model.layers:
                        layer_output = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings=(cos, sin)
                        )
                        hidden_states = layer_output[0]

                    # Final processing
                    hidden_states = model.norm(hidden_states)
                    logits = model.lm_head(hidden_states)

                    # Send result back to master
                    send_tensor(logits, dst=0)

                step_count += 1

            except Exception as e:
                log(f"Final node error: {e}")
                break

        log(f"Final node processed {step_count} steps")
        return {'steps': step_count}

    else:
        # Middle nodes: forward processing
        step_count = 0
        while True:
            try:
                # Receive from previous node
                seq_len_tensor = recv_tensor((1,), torch.int64, src=rank-1, device='cpu')
                seq_len = seq_len_tensor.item()

                if seq_len < 0:  # Stop signal
                    send_tensor(seq_len_tensor, dst=rank+1)
                    break

                # Receive tensors
                hidden_states = recv_tensor((1, seq_len, model.config.hidden_size), torch.float16, src=rank-1, device=device)
                position_ids = recv_tensor((1, seq_len), torch.int64, src=rank-1, device=device)
                cache_position = recv_tensor((seq_len,), torch.int64, src=rank-1, device=device)
                head_dim = model.config.hidden_size // model.config.num_attention_heads
                cos = recv_tensor((1, seq_len, head_dim), torch.float16, src=rank-1, device=device)
                sin = recv_tensor((1, seq_len, head_dim), torch.float16, src=rank-1, device=device)

                with torch.no_grad():
                    # Process my layers
                    for layer in model.layers:
                        layer_output = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            position_ids=position_ids,
                            past_key_value=None,
                            output_attentions=False,
                            use_cache=False,
                            cache_position=cache_position,
                            position_embeddings=(cos, sin)
                        )
                        hidden_states = layer_output[0]

                # Forward to next node
                send_tensor(seq_len_tensor, dst=rank+1)
                send_tensor(hidden_states, dst=rank+1)
                send_tensor(position_ids, dst=rank+1)
                send_tensor(cache_position, dst=rank+1)
                send_tensor(cos, dst=rank+1)
                send_tensor(sin, dst=rank+1)

                step_count += 1

            except Exception as e:
                log(f"Middle node error: {e}")
                break

        log(f"Middle node {rank} processed {step_count} steps")
        return {'steps': step_count}

def main():
    rank, world_size = setup_distributed()

    if rank is None or world_size != 4:
        log(f"Expected 4 nodes, got {world_size}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # Load model
    model = FourNodeModel(MODEL_PATH, rank, world_size, device)

    # Sync after loading
    dist.barrier()
    log("All nodes ready")

    # Test prompts
    test_prompts = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning simply.",
        "What is distributed computing?",
        "How do transformers work?"
    ]

    if rank == 0:
        print("\n" + "="*80)
        print("4-NODE LAYER SPLITTING TEST")
        print("="*80)
        print(f"Model: LLaMA 3.2-3B")
        print(f"Layer distribution:")
        for i, (start, end) in enumerate(model.layer_assignments):
            components = []
            if i == 0:
                components.append("embeddings")
            if i == world_size - 1:
                components.append("norm + lm_head")
            component_str = " + " + " + ".join(components) if components else ""
            print(f"  Node {i}: layers {start}-{end-1} ({end-start} layers){component_str}")
        print("="*80)

        results = []
        total_start = time.time()

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"Prompt: {prompt}")
            print("Response: ", end="")

            result = generate_response(prompt, model, rank, world_size, device)
            if result:
                results.append(result)

            time.sleep(1)

        total_time = time.time() - total_start
        total_tokens = sum(r['tokens'] for r in results)
        total_gen_time = sum(r['time'] for r in results)

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Prompts: {len(results)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Avg speed: {total_tokens/total_gen_time:.2f} tok/s")
        print(f"Total time: {total_time:.2f}s")
        print("="*80)

    else:
        # Worker processes all prompts
        total_steps = 0
        for i in range(len(test_prompts)):
            result = generate_response(None, model, rank, world_size, device)
            if result:
                total_steps += result['steps']

        log(f"Node {rank} completed {total_steps} steps")

    # Cleanup
    try:
        dist.barrier()
        dist.destroy_process_group()
    except:
        pass

    log("Test completed")

if __name__ == "__main__":
    main()