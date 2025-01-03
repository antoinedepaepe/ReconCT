import torch
from src.operators.operator import Operator

class DataFidelity:

    def __init__(self, forward_operator: Operator):
        self.forward_operator = forward_operator

    def loss(self, x: torch.Tensor, 
                   y: torch.Tensor, 
                   weights: torch.Tensor) -> torch.Tensor:
        
        Ax = self.forward_operator.transform(x)
        l = torch.sum( ((Ax - y)**2) * weights)
        return l
        