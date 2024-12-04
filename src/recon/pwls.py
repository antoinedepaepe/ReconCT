from src.operators.sparsifier import Sparsifier
from src.operators.operator import Operator
import torch

class PWLS:

#    def __init__(self, radon: Radon,
#                       sparsifier: TotalVariation):
    def __init__(self, systmatrix: Operator,
                          sparsifier: Sparsifier):   
       self.systmatrix = systmatrix
       self.sparsifier = sparsifier

    def solve(self,
             x0: torch.Tensor,
             b: torch.Tensor,
             beta: float,
             n_iter: int,
             weights: torch.Tensor) -> torch.Tensor:
       
       xk = x0
     
       ones = torch.ones_like(xk, device = 'cuda')
       D_rec = self.AT( weights * self.A(ones))
       D_reg = self.RabsT(self.Rabs(ones))
    
       for k in range(n_iter):
           x_rec = xk - self.AT( weights * (self.A(xk) - b)) / D_rec
           x_reg = xk - self.RT(self.R(xk)) / D_reg
           xk = (D_rec * x_rec + beta * (D_reg * x_reg))/(D_rec + beta * D_reg) ;
           xk[xk < 0] = 0
    
       return xk
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
          return self.systmatrix.transform(x)
      
    def AT(self, x: torch.Tensor) -> torch.Tensor:
        return self.systmatrix.transposed_transform(x)
      
  #  def A(self, x: torch.Tensor) -> torch.Tensor:
  #      return self.radon.transform(x)
            
  #  def AT(self, x: torch.Tensor) -> torch.Tensor:
  #      return self.radon.transposed_transform(x)

    def R(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transform(x)
          
    def RT(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transposed_transform(x)
    
    def Rabs(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transform_abs(x)
          
    def RabsT(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsifier.transposed_transform_abs(x)
