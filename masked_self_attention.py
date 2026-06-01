import torch
import torch.nn as nn
import torch.nn.functional as F # for softmax

class MaskedSelfAttention(nn.modules):
    def __init__(self, d_model =2, row_dim =0, col_dim=1):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False) #in the case of the test in the all g

