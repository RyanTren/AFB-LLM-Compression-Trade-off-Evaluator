# Training Breakdown

## 🔧 Section 1: Environment Setup & Imports
```python
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
```
#### What it does:

* Sets matplotlib's configuration directory to /tmp/mplconfig
* setdefault only sets it if it's not already set

#### Why it matters:

* On shared systems (like your VM), matplotlib tries to write config files to ~/.config/matplotlib
* If that directory doesn't exist or has permission issues, matplotlib crashes
* By pointing it to /tmp, you avoid permission problems
* This is especially important when running headless (no GUI) on servers


```python
import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from datetime import timedelta
from tqdm import tqdm
```
#### Library breakdown:

|Library|Purpose|Example Use|
|-------|-----|-----------|
|argparse|Parse command-line arguments|--epochs 3|
|time|Track training duration|Calculate total time|
|json|Save metrics to file|Export results|
|torch|PyTorch deep learning|Model training|
|matplotlib|Generate loss plots|Visualize training|
|transformersHugging| Face models|Load GPT-2 models|
|peft| Parameter-efficient fine-tuning| Apply LoRA adapters|
|datasets|Load training data|CodeParrot| 
|dataset|accelerateMulti-GPU/distributed training|Handle GPU coordination|
|DataLoader|Batch data efficiently|Feed data to model|
|IterableDatasetStream| large datasets|Avoid loading all data into RAM|
|timedelta|Format time durations|"2:15:30" instead of "8130 seconds"|
|tqdm|Progress bars|Visual feedback during training|

## 📋 Section 2: Argument Parser

```python
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="codeparrot/codeparrot-small")
    p.add_argument("--output_dir", type=str, default="lora_out_trainrun_4")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--dataset", type=str, default="codeparrot",
                   choices=["synthetic", "codeparrot"])
    p.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N steps (0=disabled)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Resume training from checkpoint directory")
    p.add_argument("--keep_last_n_checkpoints", type=int, default=3,
                   help="Keep only the last N checkpoints to save disk space")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()
```

##### Key parameters explained:
#### Training Hyperparameters:

* ``--epochs``: How many times to iterate through the entire dataset
    * ``epochs=3`` means you see all data 3 times


* ``--batch_size``: How many samples to process at once
    * Larger = faster but uses more GPU memory
    * Your M40 has 12GB, so ``batch_size=1-2`` is safe


* ``--gradient_accumulation``: Simulate larger batches without more memory
    * Effective **batch size** = batch_size × gradient_accumulation
    * Example: ``batch_size=1`` + ``gradient_accumulation=8`` = effective batch of 8
    * Gradients accumulate over 8 steps, then one optimizer update happens


* ``--learning_rate``: How big each training step is
    * ``5e-5`` = 0.00005 (common for fine-tuning)
    * Too high = training explodes
    * Too low = training is too slow



#### Model & Data:

--model_id: Which model to fine-tune (from Hugging Face Hub)
--dataset: Which dataset to use

codeparrot: Real Python code dataset
synthetic: Tiny test dataset (2 samples)



### Checkpointing:

--save_every: Save checkpoint every N optimizer steps

save_every=500 = save at steps 500, 1000, 1500...
save_every=0 = disable periodic saves


--resume_from: Path to checkpoint to continue training from

Useful if training crashes or you want to train longer


--keep_last_n_checkpoints: Delete old checkpoints to save disk space

keep_last_n=3 keeps only the 3 most recent checkpoints



#### Utility:

* ``--dry_run``: Quick test mode
    * Only processes 500 samples
    * Stops after 200 steps
    * Good for debugging




## 💾 Section 3: Checkpoint Management
### 3a. Save Checkpoint
```python
def save_checkpoint(accelerator, model, tokenizer, optimizer, epoch, step, output_dir, 
                   metrics, keep_last_n=3):
    """Save training checkpoint with state"""
    if not accelerator.is_main_process:
        return
```        
Multi-GPU coordination:

accelerator.is_main_process checks if this is the "main" GPU (rank 0)
Only the main GPU saves checkpoints to avoid duplicate writes
Other GPUs skip this function entirely

```python
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}-step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

**Creates directory like:**
```
lora_out/
  checkpoint-epoch0-step500/
  checkpoint-epoch0-step1000/
  checkpoint-epoch1-step1500/
```
```  
python    # Save model
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
```    
#### Model saving:

* ``unwrap_model``: Removes Accelerate's distributed training wrapper
* ``save_pretrained``: Saves model weights in Hugging Face format
* ``safe_serialization=True``: Uses SafeTensors format (safer than pickle)
* Tokenizer is saved too (so you can load the model independently)

```
python    # Save training state
    state = {
        "epoch": epoch,
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
```

#### Training state:

* ``epoch`` & ``step``: Where you were in training
    * ``optimizer_state``: Adam's momentum/variance buffers
        * Critical for resuming training smoothly
        * Without this, training would "forget" its momentum


* ``metrics``: Loss history, token counts, etc.

#### Why this matters:

* If training crashes at step 2500, you can resume from step 2000
* You don't lose hours of GPU time
* Optimizer state ensures resumed training continues smoothly

```
python
    cleanup_old_checkpoints(output_dir, keep_last_n)
```

Automatic cleanup to prevent disk from filling up.

### 3b. Cleanup Old Checkpoints
```
python
def cleanup_old_checkpoints(output_dir, keep_last_n):
    """Remove old checkpoints to save disk space"""
    if keep_last_n <= 0:
        return
    
    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
```
**Finds all checkpoint directories:**

* Filters for folders starting with "checkpoint-"
* Ignores files and other directories

```
python    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by modification time
    checkpoints = sorted(checkpoints, 
                        key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
```
#### Sorts by age:

* ``os.path.getmtime()`` gets last modification time
* Oldest checkpoints come first in the sorted list

```
python    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for ckpt in to_remove:
        ckpt_path = os.path.join(output_dir, ckpt)
        import shutil
        shutil.rmtree(ckpt_path)
        print(f"🗑️  Removed old checkpoint: {ckpt}")
```

**Keeps only the last N:**
- `checkpoints[:-keep_last_n]` = all except the last N
- `shutil.rmtree()` deletes entire directory tree
- Prevents disk from filling up (each checkpoint ~500MB for codeparrot-small)

**Example with keep_last_n=3:**
```
Before:
  checkpoint-epoch0-step500/   ← DELETE (oldest)
  checkpoint-epoch0-step1000/  ← DELETE
  checkpoint-epoch0-step1500/  ← KEEP
  checkpoint-epoch0-step2000/  ← KEEP
  checkpoint-epoch1-step2500/  ← KEEP (newest)

After:
  checkpoint-epoch0-step1500/
  checkpoint-epoch0-step2000/
  checkpoint-epoch1-step2500/
```

### 3c. Load Checkpoint
```
python
def load_checkpoint(checkpoint_dir):
    """Load training checkpoint"""
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path):
        return None
    
    state = torch.load(state_path)
    print(f"📂 Loaded checkpoint from epoch {state['epoch']}, step {state['step']}")
    return state
```
#### What it loads:

* Reads training_state.pt file
* Returns dictionary with epoch, step, optimizer state, and metrics
* Returns None if checkpoint doesn't exist (graceful failure)

#### How it's used:
```
pythonif args.resume_from:
    checkpoint_state = load_checkpoint(args.resume_from)
    if checkpoint_state:
        optimizer.load_state_dict(checkpoint_state["optimizer_state"])  # Restore Adam state
        start_epoch = checkpoint_state["epoch"]  # Resume from this epoch
        start_step = checkpoint_state["step"]    # Resume from this step
```

## 🚀 Section 4: Main Training Function - Part 1 (Setup)
```
python
def main():
    args = parse_args()
    start_time = time.time()
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
```

#### Initialization:

* Parse command-line arguments
* Start timer for total training time
* ``Accelerator()``: Hugging Face Accelerate's magic wrapper

* Automatically detects GPUs
* Handles distributed training
* Makes multi-GPU code look like single-GPU code


* ``is_main``: Boolean flag for "am I the main process?"

Used to prevent duplicate prints/saves


```
python    if is_main:
        print(f"🔹 Using model: {args.model_id}")
        print(f"🔹 Dataset: {args.dataset}")
        print(f"🔹 Output directory: {args.output_dir}")
        print(f"🔹 Dry run: {args.dry_run}")
        if args.resume_from:
            print(f"🔹 Resuming from: {args.resume_from}")
```
Only main process prints to avoid log spam in multi-GPU scenarios.

#### Load Tokenizer

```
python    
tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
```

**Tokenizer:**
- Converts text → numbers (token IDs) that the model understands
- `use_fast=True`: Uses Rust-based fast tokenizer (10x faster)

**Padding token fix:**
- GPT-2 doesn't have a padding token by default
- Padding is needed when batching sequences of different lengths
- We add `<|pad|>` as a special token
- Example:
```
  Sequence 1: [15, 23, 45, 67, 89]
  Sequence 2: [12, 34]  → padded to [12, 34, <PAD>, <PAD>, <PAD>]

Load Model
python    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,  # FP16 for Tesla M40
        trust_remote_code=True,
        use_safetensors=True,
    )
Model loading parameters:

torch_dtype=torch.float16: Load in half-precision (FP16)

Reduces memory usage by 50%
Codeparrot-small: 228M params × 4 bytes (FP32) = ~900MB → ~450MB (FP16)
Important: Changed from bfloat16 because M40 doesn't support it


trust_remote_code=True: Allows custom model code from Hugging Face

Some models include custom modeling code
Security consideration: only use trusted models


use_safetensors=True: Use SafeTensors format

Safer than pickle (no arbitrary code execution)
Faster loading



python    model.gradient_checkpointing_enable()
Gradient checkpointing:

Trade compute for memory
Instead of storing all intermediate activations (for backprop), recompute them
Effect: ~50% memory savings, ~20% slower training
Why it matters: Lets you fit larger models or bigger batches on your 12GB M40

python    model.resize_token_embeddings(len(tokenizer))
Resize embeddings:

We added <|pad|> token to tokenizer
Model needs an embedding vector for this new token
This adds one row to the embedding matrix


LoRA Configuration
python    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
```

**LoRA parameters explained:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r` | 8 | Rank of LoRA matrices (lower = fewer params) |
| `lora_alpha` | 32 | Scaling factor (controls adaptation strength) |
| `target_modules` | `["c_attn", "c_proj"]` | Which layers to add LoRA to |
| `lora_dropout` | 0.05 | Dropout rate (5% regularization) |
| `bias` | "none" | Don't train bias terms |
| `task_type` | "CAUSAL_LM" | Task is causal language modeling |

**What LoRA does:**

Original weight matrix **W** (e.g., 768×768):
```
Output = Input × W
```

With LoRA:
```
Output = Input × (W + A × B)
where:

W is frozen (not trained)
A is 768×8 (trainable)
B is 8×768 (trainable)

Memory savings:

Original: Train all 768×768 = 589,824 params
LoRA: Train only (768×8 + 8×768) = 12,288 params
~98% parameter reduction!

python    if is_main:
        model.print_trainable_parameters()
```

**Prints something like:**
```
trainable params: 294,912 || all params: 82,125,312 || trainable%: 0.36%
This shows you're only training 0.36% of the model!