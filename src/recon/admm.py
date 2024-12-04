from src.operators.sparsifier import Sparsifier
from src.operators.operator import Operator
import torch

def soft_thresholding(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.maximum(x.abs() - beta, torch.tensor(0).cuda())


class ADMM:

    def __init__(self, systmatrix: Operator,
                          sparsifier: Sparsifier):   
       
       self.systmatrix = systmatrix
       self.sparsifier = sparsifier

    def solve(self,
              x0: torch.Tensor,
              b: torch.Tensor,
              beta: float,
              rho: float,
              n_iter: int,
              n_inner_iter: int,
              weights: torch.Tensor):
        
        xk = x0.clone()
        zk = self.R(xk)
        uk = torch.zeros_like(zk).cuda()

        beta = torch.tensor(beta).cuda()

        ones = torch.ones_like(x0, device = 'cuda')
        D_rec = self.AT( weights * self.A(ones))
        D_reg = self.RabsT(self.Rabs(ones))

        for k in range(n_iter):
            
            # first proximal
            for _ in range(n_inner_iter):
                x_rec = xk - self.AT( weights * (self.A(xk) - b)) / D_rec
                x_reg = xk - self.RT( self.R(xk) - (zk - uk)) / D_reg
                xk = (D_rec * x_rec + rho * D_reg * x_reg) / ( D_rec + rho * D_reg) 
                xk[xk < 0] = 0

            # second proximal
            zk = soft_thresholding(self.R(xk) + uk, beta / rho)
            uk = uk + self.R(xk) - zk

        return xk
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
          return self.systmatrix.transform(x)
      
    def AT(self, x: torch.Tensor) -> torch.Tensor:
        return self.systmatrix.transposed_transform(x)

    def R(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transform(x)
          
    def RT(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transposed_transform(x)
    
    def Rabs(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transform_abs(x)
          
    def RabsT(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transposed_transform_abs(x)




