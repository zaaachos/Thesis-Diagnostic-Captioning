import os
import numpy as np
from tqdm import tqdm
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from transformers import AutoTokenizer, AutoModel
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import nltk
import torch


class ClinicalBERT:
    def __init__(self, max_length:int=40):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_length
            

    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)
        
        seq_out = self.bert_model(inputs_tensor, masks_tensor)[0]

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()
        