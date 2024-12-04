import torch
from math import sqrt
from src.operators.operator import Operator


class TotalVariation(Operator):

    def __init__(self, penalty: str = 'l1'):


        if penalty == 'l1':
            self.diag_weights = sqrt(1/2)
        elif penalty == 'l2':
            self.diag_weights = sqrt(sqrt(1/2))


    def transform(self, x: torch.Tensor, factor: float = -1.0) -> torch.Tensor:
        
        x1 = (x + factor * torch.roll(x, shifts=(0, 1), dims=(2, 3))) 
        x2 = (x + factor * torch.roll(x, shifts=(0, -1), dims=(2, 3))) 
        x3 = (x + factor * torch.roll(x, shifts=(1, 0), dims=(2, 3))) 
        x4 = (x + factor * torch.roll(x, shifts=(-1, 0), dims=(2, 3))) 
        
        x5 = self.diag_weights * (x + factor * torch.roll(x, shifts=(1, -1), dims=(2, 3))) 
        x6 = self.diag_weights * (x + factor * torch.roll(x, shifts=(-1, 1), dims=(2, 3))) 
        x7 = self.diag_weights * (x + factor * torch.roll(x, shifts=(1, 1), dims=(2, 3))) 
        x8 = self.diag_weights * (x + factor * torch.roll(x, shifts=(-1, -1), dims=(2, 3))) 
        
        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
    
    def transposed_transform(self, x: torch.Tensor, factor: float = -1.0)-> torch.Tensor:
        
        x1 = (x[:,0:1,:,:] + factor * torch.roll(x[:,0:1,:,:], shifts=(0, -1), dims=(2, 3)))
        x2 = (x[:,1:2,:,:] + factor * torch.roll(x[:,1:2,:,:], shifts=(0, 1), dims=(2, 3)))
        x3 = (x[:,2:3,:,:] + factor * torch.roll(x[:,2:3,:,:], shifts=(-1, 0), dims=(2, 3)))
        x4 = (x[:,3:4,:,:] + factor * torch.roll(x[:,3:4,:,:], shifts=(1, 0), dims=(2, 3)))
        
        x5 = self.diag_weights * (x[:,4:5,:,:] + factor * torch.roll(x[:,4:5,:,:], shifts=(-1, 1), dims=(2, 3)))
        x6 = self.diag_weights * (x[:,5:6,:,:] + factor * torch.roll(x[:,5:6,:,:], shifts=(1, -1), dims=(2, 3)))
        x7 = self.diag_weights * (x[:,6:7,:,:] + factor * torch.roll(x[:,6:7,:,:], shifts=(-1, -1), dims=(2, 3)))
        x8 = self.diag_weights * (x[:,7:8,:,:] + factor * torch.roll(x[:,7:8,:,:], shifts=(1, 1), dims=(2, 3)))
        
        res = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

        return res
    
    def norm(self, M : int, N : int) -> float:
        x, s = torch.rand(1,1, M, N), 0
        for _ in range(50):
            x, s = self.norm_one_step(x, s)
        return s

    def norm_one_step(self, x, s):
        x = self.transposed_transform(self.transform(x))
        x = x / torch.norm(x)
        s = torch.norm(self.transform(x))
        return x, s