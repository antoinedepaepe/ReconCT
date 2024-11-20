from src.operators.radon import Radon
from src.operators.total_variation import TotalVariation
import torch

class PWLS:

    def __init__(self, radon: Radon,
                       regularizer: TotalVariation):
        
        self.radon = radon
        self.regularizer = regularizer

    @torch.no_grad()
    def solve(self,
              x0: torch.Tensor,
              b: torch.Tensor,
              beta: float,
              n_iter: int,
              weights: torch.Tensor):
        
        xk = x0
  
        ones = torch.ones_like(xk, device = 'cuda')
        D_rec = self.AT( weights * self.A(ones))
        D_reg = self.RT(self.R(ones, 1.0), 1.0)

        for k in range(n_iter):
            x_rec = xk - self.AT( weights * (self.A(xk) - b)) / D_rec
            x_reg = xk - self.RT(self.R(xk)) / D_reg
            xk = (D_rec * x_rec + beta * (D_reg * x_reg))/(D_rec + beta * D_reg) ;
            xk[xk < 0] = 0

        return xk
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.radon.transform(x)
            
    def AT(self, x: torch.Tensor) -> torch.Tensor:
        return self.radon.transposed_transform(x)

    def R(self, x: torch.Tensor, factor: float = -1.0) -> torch.Tensor:
        return self.regularizer.transform(x, factor)
            
    def RT(self, x: torch.Tensor, factor: float = -1.0) -> torch.Tensor:
        return self.regularizer.transposed_transform(x, factor)
