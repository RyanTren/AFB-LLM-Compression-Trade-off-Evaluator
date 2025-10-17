# Training Run 1

This was test run one, here's everything I ran in the vm to get here

```bash
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ accelerate launch --mixed_precision "fp16" --num_processes 1 scripts/train_lora.py \
  --model_id gpt2 \
  --dataset codeparrot \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation 4 \
  --max_length 256 \
  --learning_rate 1e-4 \
  --save_every 2000 \
  --output_dir lora_out_codeparrot
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_machines` was set to a value of `1`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Using model: gpt2
🔹 Dataset: codeparrot
`torch_dtype` is deprecated! Use `dtype` instead!
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/peft/tuners/lora/layer.py:1119: UserWarning: fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True.
  warnings.warn(
trainable params: 294,912 || all params: 124,735,488 || trainable%: 0.2364
📘 Loading CodeParrot (subset 1%)...
Resolving data files: 100%|██████████████████████████████████████████████████████████████| 54/54 [00:00<00:00, 22111.92it/s]
Map: 100%|████████████████████████████████████████████████████████████████████| 53614/53614 [01:36<00:00, 556.79 examples/s]
🚀 Starting LoRA fine-tuning...
Epoch 0:   0%|                                                                                    | 0/26807 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
Epoch 0:   7%|███▋                                             | 2000/26807 [10:53<2:15:40,  3.05it/s, smoothed_loss=0.4660]/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/peft/utils/save_and_load.py:209: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step2000
Epoch 0:  15%|███████▎                                         | 4000/26807 [21:50<2:04:25,  3.06it/s, smoothed_loss=0.5461]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step4000
Epoch 0:  22%|██████████▉                                      | 6000/26807 [32:46<1:53:58,  3.04it/s, smoothed_loss=0.4297]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step6000
Epoch 0:  30%|██████████████▌                                  | 8000/26807 [43:43<1:42:40,  3.05it/s, smoothed_loss=0.4914]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step8000
Epoch 0:  37%|█████████████████▉                              | 10000/26807 [54:40<1:32:04,  3.04it/s, smoothed_loss=0.4698]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step10000
Epoch 0:  45%|████████████████████▌                         | 12000/26807 [1:05:37<1:20:58,  3.05it/s, smoothed_loss=0.4857]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step12000
Epoch 0:  52%|████████████████████████                      | 14000/26807 [1:16:34<1:09:59,  3.05it/s, smoothed_loss=0.5147]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step14000
Epoch 0:  60%|████████████████████████████▋                   | 16000/26807 [1:27:31<59:09,  3.04it/s, smoothed_loss=0.4901]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step16000
Epoch 0:  67%|████████████████████████████████▏               | 18000/26807 [1:38:28<48:09,  3.05it/s, smoothed_loss=0.4329]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step18000
Epoch 0:  75%|███████████████████████████████████▊            | 20000/26807 [1:49:25<37:17,  3.04it/s, smoothed_loss=0.4614]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step20000
Epoch 0:  82%|███████████████████████████████████████▍        | 22000/26807 [2:00:24<26:15,  3.05it/s, smoothed_loss=0.4773]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step22000
Epoch 0:  90%|██████████████████████████████████████████▉     | 24000/26807 [2:11:22<15:21,  3.05it/s, smoothed_loss=0.4368]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step24000
Epoch 0:  97%|██████████████████████████████████████████████▌ | 26000/26807 [2:22:21<04:26,  3.03it/s, smoothed_loss=0.4438]💾 Saved intermediate checkpoint: lora_out_codeparrot/checkpoint_step26000
Epoch 0: 100%|████████████████████████████████████████████████| 26807/26807 [2:26:47<00:00,  3.04it/s, smoothed_loss=0.4436]
✅ Epoch 0 complete | Avg loss: 1.9380
💾 Epoch 0 checkpoint saved to lora_out_codeparrot/checkpoint_epoch0
📊 Metrics saved to: lora_out_codeparrot/metrics_1760728470.json
📈 Loss plot saved to: lora_out_codeparrot/loss_plot_1760728470.png

✅ Training complete! LoRA adapters + tokenizer saved to: lora_out_codeparrot
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git add scripts/train_lora.py requirements.txt lora_out_codeparrot/*.json lora_out_codeparrot/*.png .gitignore
fatal: pathspec '.gitignore' did not match any files
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git add scripts/train_lora.py requirements.txt lora_out_codeparrot/*.json lora_out_codeparrot/*.png
The following paths are ignored by one of your .gitignore files:
src/lora_out_codeparrot
hint: Use -f if you really want to add them.
hint: Turn this message off by running
hint: "git config advice.addIgnoredFile false"
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "venv/" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "__pycache__/" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "*.bin" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "lora_out_codeparrot/checkpoint_*" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "lora_out_codeparrot/pytorch_model*" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ echo "lora_out_codeparrot/adapter_model.safetensors" >> .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ cat .gitignore
venv/
__pycache__/
*.bin
lora_out_codeparrot/checkpoint_*
lora_out_codeparrot/pytorch_model*
lora_out_codeparrot/adapter_model.safetensors
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git add scripts/train_lora.py requirements.txt .gitignore
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git add lora_out_codeparrot/*.json lora_out_codeparrot/*.png
The following paths are ignored by one of your .gitignore files:
src/lora_out_codeparrot
hint: Use -f if you really want to add them.
hint: Turn this message off by running
hint: "git config advice.addIgnoredFile false"
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git add -f lora_out_codeparrot/*.json lora_out_codeparrot/*.png
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git commit -m "Add LoRA fine-tuning script, metrics, and loss plot for CodeParrot"
git push
Author identity unknown

*** Please tell me who you are.

Run

  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"

to set your account's default identity.
Omit --global to set the identity only in this repository.

fatal: empty ident name (for <p10-t1llmcomp@GPU2.maas>) not allowed
Everything up-to-date
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git config -global user.email "concepting@protonmail.com"
error: did you mean `--global` (with two dashes)?
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git config --global user.email "concepting@protonmail.com"
error: could not lock config file /mnt/sperry46/p10-t1llmcomp/.gitconfig: No such file or directory
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git config --global user.name "RyanTren"
error: could not lock config file /mnt/sperry46/p10-t1llmcomp/.gitconfig: No such file or directory
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ # Set git identity for this repository only (no --global flag)
git config user.email "concepting@protonmail.com"
git config user.name "RyanTren"

# Verify the configuration
git config user.email
git config user.name

# Now retry the commit
git commit -m "Add LoRA fine-tuning script, metrics, and loss plot for CodeParrot"
git push
concepting@protonmail.com
RyanTren
[lora 00df265] Add LoRA fine-tuning script, metrics, and loss plot for CodeParrot
 10 files changed, 250508 insertions(+), 37 deletions(-)
 create mode 100644 src/.gitignore
 create mode 100644 src/lora_out_codeparrot/adapter_config.json
 create mode 100644 src/lora_out_codeparrot/added_tokens.json
 create mode 100644 src/lora_out_codeparrot/loss_plot_1760728470.png
 create mode 100644 src/lora_out_codeparrot/metrics_1760728470.json
 create mode 100644 src/lora_out_codeparrot/special_tokens_map.json
 create mode 100644 src/lora_out_codeparrot/tokenizer.json
 create mode 100644 src/lora_out_codeparrot/tokenizer_config.json
 create mode 100644 src/lora_out_codeparrot/vocab.json
Enumerating objects: 17, done.
Counting objects: 100% (17/17), done.
Delta compression using up to 48 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (14/14), 975.39 KiB | 4.28 MiB/s, done.
Total 14 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
To github.com:RyanTren/AFB-LLM-Compression-Trade-off-Evaluator.git
   b555ccb..00df265  lora -> lora
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ ^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ python scripts/run_inference.py
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Loading GPT-2 + LoRA adapter...
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Write a Python function that reverses a string.
Write a Python function that reverses a string.

import os import time import sys import time.sleep import time.sleep.sleep_time import time.sleep.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time.sleep_time
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Explain how a neural network learns.
Explain how a neural network learns.

The first step is to understand how the network learns. The second step is to understand how the network learns.

The first step is to understand how the network learns.

The second step is to understand how the network learns.

The third step is to understand how the network learns.

The third step is to understand how the network learns.

The fourth step is to understand how the network learns.

The fourth step is to understand how the network
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Generate a short poem about AI and the Air Force.
Generate a short poem about AI and the Air Force.

The Air Force is a major military organization. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Result:
Generate a short poem about AI and the Air Force.

The Air Force is a major military organization. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a large number of officers and enlisted personnel. It has a
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git fetch
^C
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git fetch
remote: Enumerating objects: 15, done.
remote: Counting objects: 100% (15/15), done.
remote: Compressing objects: 100% (6/6), done.
remote: Total 10 (delta 7), reused 5 (delta 4), pack-reused 0 (from 0)
Unpacking objects: 100% (10/10), 4.28 KiB | 1.43 MiB/s, done.
From github.com:RyanTren/AFB-LLM-Compression-Trade-off-Evaluator
   00df265..1ba43cc  lora       -> origin/lora
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ git pull
Updating 00df265..1ba43cc
Fast-forward
 src/lora_out_codeparrot/README.md | 147 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/scripts/run_inference.py      |   2 +-
 2 files changed, 148 insertions(+), 1 deletion(-)
 create mode 100644 src/lora_out_codeparrot/README.md
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ python scripts/run_inference.py
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Loading GPT-2 + LoRA adapter...
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Write a Python function that reverses a string.
Write a Python function that reverses a string.

import os
import re
import re
import re.compile
import re.log
import re.log.import_function
import re.log.import_string
import re.log.import_string.from_string
import re.log.import_string.from_string.from_string
import re.log.import_string.from_string.from_string.from_string.from_string.from_string.from
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Explain how a neural network learns.
Explain how a neural network learns.

import os
import re
import re
import re.compile
import re.log
import re.log.import_module
import re.log.import_string
import re.log.import_string.from_string
import re.log.import_string.from_string.from_string
import re.log.import_string.from_string.from_string.from_string.from_string.from_string.from
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Prompt: Generate a short poem about AI and the Air Force.
Generate a short poem about AI and the Air Force.

"""

from __future__ import unicode_literals

import os
import re
import re.compile
import re.log
import re.version
import re.version.extensions
import re.version.extensions.extensions.extension_types
import re.version.extensions.extensions.extension_types.extension_types.extension_types.extension_types.extension_types.ext
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

🔹 Result:
Generate a short poem about AI and the Air Force.

"""

from __future__ import unicode_literals

import os
import re
import re.compile
import re.log
import re.version
import re.version.extensions
import re.version.extensions.extensions.extension_types
import re.version.extensions.extensions.extension_types.ext

```
