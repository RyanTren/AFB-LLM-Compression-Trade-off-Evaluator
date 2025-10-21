# Results/Test Runs

In this project the models we used for LoRA/PEFT was GPT2, CodeParrot-Small, and Salesfoce/CodeGen. In [Training Run 1](C:\Users\ryant\OneDrive\Documents\GitHub\AFB-LLM-Compression-Trade-off-Evaluator\src\lora_out_codeparrot) and [Training Run 2](src/lora_out_codegen_final) we used GPT2 and the results from training and running inference scripts on our model gave us disapointing outputs for code generation and most of the time was spent training the model on a small JSON dataset and eventual a codeparrot dataset with over 100M lines of training data.



## Training Run 1

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

# Training Run 2

Some improvement logs here:
``train_lora.py``
* uses CodeParrot in streaming mode (no full dataset in RAM)
* trains efficiently with LoRA + FP16
* saves periodic checkpoints
* will handle multi-hour runs without memory leaks

``Launch Command``
```bash
accelerate launch --mixed_precision "fp16" --num_processes 1 scripts/train_lora.py \
  --model_id gpt2 \
  --dataset codeparrot \
  --epochs 1 \
  --batch_size 2 \
  --gradient_accumulation 8 \
  --max_length 512 \
  --learning_rate 1e-4 \
  --save_every 5000 \
  --output_dir lora_out_codegen
```

## Results

```bash
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src$ python scripts/run_inference.py
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Using device: cuda
🔹 Loading GPT-2 + LoRA adapter from: lora_out_codegen_final
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
✅ Model loaded successfully.
🧠 Loaded 5 prompts for category 'string_algorithms'.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

================================================================================
📝 Prompt 1: # Reverse a string in Python.
--------------------------------------------------------------------------------
# Task: # Reverse a string in Python.
# Solution:
# ============================================================================
#
# Copyright (C) 2008-2010, Jelte, Stefan
# Author: Stefan Eriksson <lj@jelte.net>
# ============================================================================
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

================================================================================
📝 Prompt 2: # Check if a string is a palindrome.
--------------------------------------------------------------------------------
# Task: # Check if a string is a palindrome.
# Solution:
# -*- coding: utf-8 -*-

# This program is part of the OpenOffice Platform.
# Please see https://www.openoffice.org/~kim-paul/docs/
# OpenOffice.org Documentation:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

================================================================================
📝 Prompt 3: # Count vowels and consonants in a string.
--------------------------------------------------------------------------------
# Task: # Count vowels and consonants in a string.
# Solution:
# # <a href="http://www.microsoft.com/en-us/library/windows-browser/cs7/en/html" target="_blank">http://www.microsoft.com/en-us/library/windows-browser/cs7/en/html</a>
#
# Copyright (C) 2008, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 1926, 1927, 1928, 1929, 1931, 1932, 1933, 1934, 1935, 1936
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

================================================================================
📝 Prompt 4: # Remove duplicate characters from a string.
--------------------------------------------------------------------------------
# Task: # Remove duplicate characters from a string.
# Solution:
# # Remove duplicate characters from a string.
# ################################################################################
#
#    Open Source PEP 8
#    Copyright (C) 2005, 2008, 2009, 2010, 2011
#
#    This file is part of the Open Source PEP 8 Project
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

================================================================================
📝 Prompt 5: # Compress a string using run-length encoding.
--------------------------------------------------------------------------------
# Task: # Compress a string using run-length encoding.
# Solution:
#
#    This example compresses a string using run-length encoding.
#    The following example compresses a string using run-length encoding.
#
#    The following example compresses a string using run-length encoding.
#
#
# This example compresses a string using run-length encoding.
#
#    In addition to the two examples in this example, the following example compresses a string with run-length encoding.
#
#
#

✅ Inference complete!

```

Instead of generating concise Python solutions, your model is producing large blocks of copyright notices, boilerplate comments, and repeated phrases. That’s usually a sign of overfitting or a data mismatch.

### Why the output looks like this

#### Dataset choice and formatting (codeparrot)

codeparrot is a huge Python code dataset scraped from GitHub. It contains a lot of license headers, copyright comments, and boilerplate.

If you train directly on it with low epochs, the LoRA might latch onto these patterns, especially for small prompts like # Reverse a string in Python.

Result: The model tends to generate long comment blocks instead of actual solutions.

#### Training configuration

You trained only 1 epoch with batch size 2 and gradient accumulation 8. That’s a very light training run. LoRA may not have enough signal to generalize from task-specific patterns, so it defaults to copying patterns in the dataset (license headers, boilerplate, etc.).

max_length 512 is fine, but sometimes the model might generate too many tokens and get stuck in loops of repeated text.

#### Prompt design at inference

Your prompts are short (# Reverse a string in Python.). Without clear task-specific formatting, the model leans on common patterns it saw in codeparrot.

CodeParrot has a lot of “example code” and license headers — that’s why your model keeps outputting them.

### Best improvements:

Filter the dataset for short, clean examples.

Train for more epochs with a slightly lower LR.

Use task-specific prompts with examples.

Adjust generation parameters (max_new_tokens, top_p, repetition_penalty).

# Training Run 3

```bash
(venv) p10-t1llmcomp@GPU2:/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src                                                       $ python scripts/run_inference.py
/tmp/p10-t1llmcomp/AFB-LLM-Compression-Trade-off-Evaluator/src/venv/lib/python3.12/site-                                                       packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is dep                                                       recated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔹 Using device: cuda
🔹 Loading GPT-2 + LoRA adapter from: ./lora_out_codegen_final
`torch_dtype` is deprecated! Use `dtype` instead!
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covar                         iance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mea                         n_resizing=False`
✅ Model loaded successfully.
🧠 Loaded 5 prompts for category 'string_algorithms'.

================================================================================
📝 Prompt 1: # Reverse a string in Python.
--------------------------------------------------------------------------------
if len(s) > 1 and isset(s):

"""A valid Unicode character with the following meanings (in ascending order).
Example code for an example of this value:"""
  '''' {
  name: "Tiny",
}

 """An unary ASCII letter or number that has no characters from outside its alphabet."""

================================================================================
📝 Prompt 2: # Check if a string is a palindrome.
--------------------------------------------------------------------------------
"""Check whether the character contains an underscore or unicode in this case."""
try:

import ldasl
except ImportError, Exceptions as e: // _._setfenv('undefined')
from pyc2 import CAPI
class Palindrome(CAPI): """Syntax for converting characters to hexadebimal numbers by passing them along with your inp                         ut values and then using these parameters together.''"" id = 'Palindrom'
if not isset(id):
abort=True
characters = {'a': 0}

print("A number of digits

================================================================================
📝 Prompt 3: # Count vowels and consonants in a string.
--------------------------------------------------------------------------------
for _,a=0..num_rows,p=len(s[:, 1],numbers) do
except KeyboardInterrupt: print("Not counting vowel count".format(r '^'))

"""Check for length of the index strings.", checkIndex = False

""Validate all indexes returned by fget() on this row."""
return True

 """This function returns an array containing every integer value that can be considered as part or entire result at any given time (inclusive).

"""In addition to other functions like FGet(), we also provide one more method called ``find():`` which

================================================================================
📝 Prompt 4: # Remove duplicate characters from a string.
--------------------------------------------------------------------------------
allletters = allletters in ['ch'][0].split('\w')

if len(word) > 0 and not word.lower() or term.upper(): print '%d+', str(words))

else :

 """Tests for the inverse of an English alphabet."""

 checksum += 1

     fc, cbl=True
  __doc__ = lambda x: (x + 2)/4
self._checkDict([{}, {}], checkdict(), function(_fctrpy), _exception)]
    raise Traceback

================================================================================
📝 Prompt 5: # Compress a string using run-length encoding.
--------------------------------------------------------------------------------
try :
except ValueError, exceptionHandlerNotFoundException as e:

from __future__ import division (absolute_import) except ImportError , ECONNECTIONALESOURCE = None ): """Printed unicode characters with an absolute length of 0 or greater."""
if not os.path.exists('utf8').lower() in [0 for i in range(len(e.body))]: elsef
return '\r
'

class DoublePair(object): ... def set_keyword(): print('*')

"""Set the key word to be used by double pairs that

✅ Inference complete!
```

## Thing I'll try to implement:

Use a code-capable base model — e.g. ``base_model = "Salesforce/codegen-350M-multi"`` instead of gpt2.
That will hopefully drastically improve syntax correctness.

* Fine-tune longer or with a cleaner dataset of (instruction, completion) pairs.

The accelerate scripts (if you're cd'd into /src use these commands):
```bash
accelerate launch ../src/scripts/train_lora.py \
  --model_id codeparrot/codeparrot-small \
  --output_dir lora_out_codeparrot_small \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot

```

```bash
accelerate launch ../src/scripts/train_lora.py \
  --model_id Salesforce/codegen-350M-mono \
  --output_dir lora_out_codegen_final \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --learning_rate 5e-5 \
  --max_length 128 \
  --dataset codeparrot
```

- Lower temperature in your inference:

``outputs = model.generate(inputs, temperature=0.2, max_new_tokens=128)``
(Right now it’s likely defaulting to 1.0 → high randomness.)

Add proper prompt formatting, e.g.:

##### Task: Write a Python function to reverse a string.
```python
def reverse_string(s):
```
That helps steer the generation toward functional code.


