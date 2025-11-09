import os,math,random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy import sparse
import torch.optim as optim


#parameters for distillation
LR     = 2e-4
T      = 2.0     # KD temperature
ALPHA  = 0.2     # CE weight
BETA   = 0.8     # KD weight
OUT_DIR = "/home/pranavkartha/llama3/student"

teacher_model = "/home/pranavkartha/llama3/Llama3.1-8B/consolidated.00.pth"
teacher_param = "/home/pranavkartha/llama3/Llama3.1-8B/params.json"
teacher_tokens = "/home/pranavkartha/llama3/Llama3.1-8B/tokenizer.model"

student_model = "/home/pranavkartha/llama3/Llama3.2-1B/consolidated.00.pth"
student_param = "/home/pranavkartha/llama3/Llama3.2-1B/params.json"
student_tokens = "/home/pranavkartha/llama3/Llama3.2-1B/tokenizer.model"
student_path =  "/home/pranavkartha/llama3/Llama3.2-1B"
# Forward KL divergence loss
class ForwardKLLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, labels) -> torch.Tensor:
        # Implementation from https://github.com/jongwooko/distillm
        # Computes the softmax of the teacher logits
        with torch.no_grad():
          teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        # Computes the student log softmax probabilities
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        # Computes the forward KL divergence
        prod_probs = teacher_prob * student_logprob
        # Compute the sum
        x = torch.sum(prod_probs, dim=-1).view(-1)
        # We don't want to include the ignore labels in the average
        mask = (labels != self.ignore_index).float()
        # Loss is averaged over non-ignored targets
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class Producelogit(torch.nn.Module):
  #load model
  def get_logits(self,modelpath,param,token,device,datasetpath):
      with open(param,'r') as f:
       config = json.load(f)
      model_args = ModelArgs(**config)
      model = Transformer(model_args)
      model.eval()
      checkpoint = torch.load(modelpath, map_location="cpu")
      model.load_state_dict(checkpoint, strict=False)
      #load tokenizer then make tokens
      tokenizer = Tokenizer(token)
      text = (datasetpath["question"] + " " + datasetpath["code"]).tolist()
      text = random.sample(text,6)
      encoded = []
      for t in text:
        if(isinstance(t,str)):
          x = tokenizer.encode(t, bos=True, eos=True)
          encoded.append(x)
      # Pad / truncate manually
      padded = []
      for e in encoded:
          if len(e) < 512:
              e = e[:512] + [tokenizer.pad_id] * (512 - len(e))
          else:
              e = e[:512]
          #print(e)
          padded.append(e)

      #move model to GPU
      model = model.to(dtype=torch.float16)
      model.to(device)
      #get logits from tokens
      tokens = torch.tensor(padded, dtype=torch.long, device=device)

      #print(tokens.shape)

      with torch.inference_mode():
       logits = model(tokens, start_pos=0)

      #save model temporarily
      #torch.save(model.state_dict(),"/home/pranavkartha/llama3/tempmodel/model.pth")

      return logits, tokens

#set device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#data set for training
datasetpath = pd.read_csv("/home/pranavkartha/llama3/Python codes.csv")
logits = Producelogit()
slogits, stokens = logits.get_logits(student_model,student_param,student_tokens,device,datasetpath)
'''
tlogits, ttokens = logits.get_logits(teacher_model,teacher_param,teacher_tokens,device,datasetpath)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
'''
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
loss = ForwardKLLoss()
totalloss = loss.forward(slogits,slogits,stokens)
print(totalloss)

with open(student_param, 'r') as f:
    config = json.load(f)
smodel_args = ModelArgs(**config)
smodel = Transformer(smodel_args)

dictionary = torch.load("/home/pranavkartha/llama3/tempmodel/model.pth",map_location="cpu")
smodel.load_state_dict(dictionary)
#update weights and load new model
optimizer = optim.SGD(smodel.parameters(), lr=0.01)
smodel.zero_grad()
totalloss.backward()
optimizer.step()
torch.save(smodel.load_state_dict(),"/home/pranavkartha/llama3/student/selfdistilledmodel.pth")
