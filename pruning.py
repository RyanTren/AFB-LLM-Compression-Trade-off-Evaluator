import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
from llama.model import Transformer, ModelArgs, FeedForward

# Step 1: Define model
model_args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=None,
    vocab_size=32000,
    max_batch_size=32,
    max_seq_len=2048,
)

model = Transformer(model_args)

var = input("please enter file path:")
file = torch.load(var)
model.load_state_dict(file)
model.eval()
prune.l1_unstructured(model.layers[0].feed_forward.w1, name = "weight", amount=.5)
torch.save(model, "pruning/pruned"+var)



