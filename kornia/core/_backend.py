# File containing the backend that defaults to Pytorch.
import torch

# classes
Tensor = torch.Tensor
Module = torch.nn.Module

# functions
concatenate = torch.cat
stack = torch.stack
normalize = F.normalize

# constructors
as_tensor = torch.as_tensor

# random
rand = torch.rand
