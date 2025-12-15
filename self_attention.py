import torch
import torch.nn as nn
import torch.nn.functional as F # for softmax

# in this file we just implement the class and weights are random. there is no training here!
# In self-attention, we have just one input: embedding of input text. based on 3 different weight matrices, we create q, k and v
# this formula and a code, calculate scaled dot product similarities among all of the words.
# For example if we have 10 words and 20 is embedding size, at the end we have 10 x 20 matrix so we have 20 attention score for each word
class SelfAttention(nn.Module):  # define new class inherited form nn.Module 

    def __init__(self, d_model=2, row_dim=0, col_dim=1): 
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim
    
    def forward(self, token_encoding):
        q = self.W_q(token_encoding)
        k = self.W_k(token_encoding)
        v = self.W_v(token_encoding)
        # here we have 3 series of weights. when we start to train them
        sims = torch.matmul(q,k.transpose(dim0=self.row_dim, dim1=self.col_dim))  # similarity matrix between query and keys
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim) # col_dim is 1 so it calculate softmax based on rows!
        attention_score = torch.matmul(attention_percents,v)
        print(sims.shape)
        return attention_score
# in this example, we just have the query in the size of m*n where m is embedding size and n is number of tokens in query statement
encoding_matrix = torch.tensor([[1.16, 0.23],[0.57, 1.36],[4.41, -2.16]]) # the encoded values of query which is used to calculate q,k,v matrices
# encoding_matrix = torch.tensor([[1.16, 0.23, 0.3, 0.4, 0.6],[0.57, 1.36, 0.3, 0.4, 0.6],[4.41, -2.16, -0.3, -0.4, -0.6]]) # the encoded values of query which is used to calculate q,k,v matrices

torch.manual_seed(42)
SelfAttention = SelfAttention(d_model=2, row_dim=0, col_dim=1)
print(SelfAttention(encoding_matrix))




