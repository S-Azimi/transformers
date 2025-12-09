import torch
import torch.nn as nn
import torch.nn.functional as F # for softmax


class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1): 
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim
    
    def forward(self, token_encoding):
        q = self.W_q(token_encoding)
        k = self.W_q(token_encoding)
        v = self.W_q(token_encoding)

        sims = torch.matmul(q,k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim) # col_dim is 1 so it calculate softmax based on rows!
        attention_score = torch.matmul(attention_percents,v)
        return attention_score

encoding_matrix = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41, -2.16]])

torch.manual_seed(42)
SelfAttention = SelfAttention(d_model=2, row_dim=0, col_dim=1)
print(SelfAttention(encoding_matrix))

