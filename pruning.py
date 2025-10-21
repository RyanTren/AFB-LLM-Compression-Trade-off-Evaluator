import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os

var = input("please enter file path:")
model = torch.load(var)
prune.l1_unstructured(model, name = "weight", amount=.5)
torch.save(model, "pruning/pruned"+var)



