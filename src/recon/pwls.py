from src.operators.operator import Operator
from src.operators.sparsifier import Sparsifier
from src.operators.gradient import Gradient
import torch

class PWLS:

#    def __init__(self, radon: Radon,
#                       sparsifier: TotalVariation):
    def __init__(self, systmatrix: Operator,sparsifier: Sparsifier):   
       self.systmatrix = systmatrix
       self.sparsifier = sparsifier

    def solve(self,
             x0: torch.Tensor,
             b: torch.Tensor,
             beta: float,
             n_iter: int,
             weights: torch.Tensor,
             prior: str = 'quadratic',
             delta_hub: float=0.001
             ) -> torch.Tensor:
       
       xk = x0
       
       
       if prior == 'huber':
           sparsifier_ones = Gradient(weight = 'ones')
           
     
       ones = torch.ones_like(xk, device = 'cuda')
       D_rec = self.AT( weights * self.A(ones))
       if prior =='quadratic':
        D_reg = self.RabsT(self.Rabs(ones))
    
       for k in range(n_iter):
           x_rec = xk - self.AT( weights * (self.A(xk) - b)) / D_rec
           
           if prior =='huber':
            diff = sparsifier_ones.transform(xk)
            c = self.psiprime(diff,delta_hub)/diff
            c[diff == 0] = 1
            D_reg = self.RabsT(c*self.Rabs(ones))
            x_reg = xk - self.RT(c*self.R(xk)) / D_reg
           elif prior=='quadratic':
            x_reg = xk - self.RT(self.R(xk)) / D_reg
           xk = (D_rec * x_rec + beta * (D_reg * x_reg))/(D_rec + beta * D_reg) 
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
    
    def psiprime(self,t: torch.Tensor, delta:float) -> torch.Tensor:
        return t * (torch.abs(t) <= delta) + delta * (t > delta) - delta * (t < -delta)
