## Limitations
Tesla M40 GPUs (compute capability ~5.2) are likely to run into compatibility / performance problems with modern bitsandbytes 4-bit / int8 kernels that QLoRA depends on. You can try a few approaches (compile from source, heavy CPU offload, or use smaller models + model-parallel tricks), but the cleanest routes are either (A) use newer GPUs (Ampere/RTX30/40 / A100 family) for QLoRA training or (B) do a hybrid approach (smaller model / inference-only quantization / offline GPTQ) on your M40 cluster. I’ll give you a concrete checklist + commands, Docker tips, and fallback options so you can try immediately.

- Tesla M40 has compute capability 5.2 (older Maxwell/Kepler generation), while modern bitsandbytes LLM kernels (LLM.int8 / 4-bit NF4 used by QLoRA) target newer CUDA features and higher compute capability GPUs -> this causes unsupported-kernel errors or very poor performance.
- QLoRA (the approach) relies on bitsandbytes 4-bit quantization (NF4) to keep models trainable and tiny; if bitsandbytes GPU kernels cannot load, you lose that path.

```bash
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ python scripts/cuda_compute_capability.py name: Tesla M40 major,minor: 5 2
```
Your Tesla M40 (compute capability 5.2) confirms what I suspected:
it’s a Maxwell-generation GPU (2015) — powerful for its time but too old for modern QLoRA GPU kernels.

### 🧩 What “compute capability 5.2” means

QLoRA relies on bitsandbytes (bnb) for 4-bit quantization and fused CUDA kernels.

These kernels require GPUs with compute capability ≥ 6.1 (Pascal/Volta) and are fully optimized for Ampere+ (8.0/8.6).

On 5.2 GPUs, bitsandbytes will either:

Refuse to load (“unsupported device compute capability”), or

Fall back to extremely slow CPU emulation (making it impractical).

So: ❌ no QLoRA fine-tuning on these GPUs.

### 🧠 What you can still do on these M40s
Option 1 — Do LoRA or PEFT fine-tuning without quantization

You can still:

Load a model in FP16 or 8-bit mode (if memory fits)

Apply LoRA adapters (PEFT)

Offload optimizer states and gradients to CPU with DeepSpeed ZeRO-Offload or accelerate

It’s slower, but it works on old GPUs because it doesn’t depend on bitsandbytes 4-bit kernels.

👉 This is often called “LoRA fine-tuning” (not “QLoRA” because it doesn’t use NF4 quantization).

## Pivot
🎯 Project intent (from your PDF)

“Evaluate the inference quality … of at least one open-source LLM undergoing various compression techniques (such as quantization, knowledge distillation, or anything else).”

10-LLM Compression Trade-off Ev…

That line — “or anything else” — gives you flexibility.
The project’s focus is evaluating compression trade-offs, not necessarily implementing every compression method.

✅ Why LoRA/PEFT fine-tuning qualifies

LoRA (Low-Rank Adaptation) and its generalization, PEFT, are parameter-efficient fine-tuning techniques that reduce memory and compute cost versus full fine-tuning.
They are considered a compression or efficiency technique because they:

Train and store only a few million additional parameters instead of billions.

Dramatically cut fine-tuning memory/compute requirements.

Enable multiple specialized models sharing one frozen base.

Even without quantization, LoRA still:

Reduces trainable parameter count.

Improves deployment efficiency.

Demonstrates the performance vs. size trade-off your sponsor wants studied.

So your LoRA/PEFT runs can form one axis of the compression trade-off comparison.

💡 How to align this with the project rubric
Project Requirement	How LoRA/PEFT Satisfies It
“Evaluate LLM inference quality under various compression techniques.”	Compare LoRA-fine-tuned vs. full model vs. (optionally) quantized or pruned versions.
“Analyze model compression techniques and their impact on query accuracy and speed.”	Measure speedup (smaller adapter vs. full model) and accuracy change on Code-BLEU.
“Build dynamic query allocation prototype between full and compressed models.”	Route simple queries to LoRA model and complex ones to full model.
“Visualize performance trends and resource savings.”	Plot inference latency, GPU memory use, and BLEU scores.

That’s fully compliant.

🧩 Suggested experiment plan

If you can’t run quantization yet (due to GPU limits):

Baseline: Full-precision LLM (e.g., Llama-2-7B or Falcon-1B).

Compression 1: LoRA/PEFT fine-tuned model (parameter-efficient).

Compression 2 (optional): CPU-quantized or GPTQ-compressed inference model.

Measure:

Inference latency

GPU/CPU memory footprint

Code-BLEU / BLEU / pass@k metrics

Model size (MB)

Even just comparing (1) vs (2) already demonstrates a compression-accuracy trade-off.

### 🧠 Summary

LoRA/PEFT without quantization = valid compression method.

You’ll still meet the project objectives around trade-off evaluation, performance analysis, and visualization.

Later, if you get access to newer GPUs, you can add QLoRA as an extra comparison — but it’s not required to fulfill the project.