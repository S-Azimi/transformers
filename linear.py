import torch
import torch.nn as nn
# simple linear


linear_layer = nn.Linear(in_features=3, out_features=2, bias=False)
input_tensor = torch.randn(10, 5)  # Batch size 10, 5 features

input_tensor =  torch.tensor([[1, 2, 1],[1, 2, 3]]).float()

output = linear_layer(input_tensor)

W= linear_layer.weight
print(f"output:   {output}")

o2=torch.matmul(input_tensor,W.T)
o2=torch.matmul(input_tensor,W.transpose(1,0))
print(f"manual:  {o2}")

