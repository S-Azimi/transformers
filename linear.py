import torch
import torch.nn as nn
# simple linear


linear_layer = nn.Linear(in_features=3, out_features=2)
input_tensor = torch.randn(10, 5)  # Batch size 10, 5 features

input_tensor =  torch.tensor([[1, 2, 1],[1, 2, 3]]).float()

output = linear_layer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"weight: {linear_layer.weight}")
print(output)

