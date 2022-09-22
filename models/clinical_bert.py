# os imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# progress bar and numpy imports
import numpy as np
from tqdm import tqdm

# transformers imports
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import torch


class ClinicalBERT:
    def __init__(self, max_length:int=40):
        """ ClinicalBERT class that imports the pre-trained model from net.
        It employs the transformers libraries.

        Args:
            max_length (int, optional): The max sequence length to be used in order to cut or pad the captions. Defaults to 40, after my preliminary experiments.
        """
        # send all variables in GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        # official ClinicalBERT model
        self.model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_length
            

    def vectorize(self, sentence : str) -> np.array:
        """ Takes a sentence as input and it converts into text embeddings using the pre-trained ClinicalBERT model.
        It is effective only on medical captions as the ClinicalBERT is trained on numerous medical texts.

        Args:
            sentence (str): The sentence (caption) to be vectorized.

        Returns:
            np.array: The text embeddings into np.array format to conform with Tensorflow. 
        """
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        # if caption is longer than 40, cut.
        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            # else add pad tokens
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        # send variables to GPU if available
        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)
        
        # extract text embeddings using pre-trained model
        seq_out = self.bert_model(inputs_tensor, masks_tensor)[0]

        # convert torch tensors to numpy arrays and return
        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()
        