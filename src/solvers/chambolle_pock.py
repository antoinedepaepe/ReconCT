from src.operators.radon import Radon
from src.operators.total_variation import TotalVariation
import torch




# compute first proximal prox_sigma_h*

def soft_thresholding(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.minimum(x.abs(), beta)




import matplotlib.pyplot as plt

class ChambollePock:

    def __init__(self, radon: Radon,
                       regularizer: TotalVariation):
        
        self.radon = radon
        self.regularizer = regularizer

    @torch.no_grad()
    def solve(self,
              x0: torch.Tensor,
              b: torch.Tensor,
              beta: float,
              tau: float,
              sigma: float,
              theta: float,
              n_iter: int,
              n_inner_iter: int,
              weights: torch.Tensor):
        
        # if not weights:
        #     weights = torch.ones_like(b, device = 'cuda')

        xk = x0.clone()
        xbar = x0.clone()
        z = torch.zeros_like(self.R(xbar)).cuda()

        beta = torch.tensor(beta).cuda()
  

        ones = torch.ones_like(x0, device = 'cuda')
        D_rec = self.AT( weights * self.A(ones))

        for k in range(n_iter):

            # first proximal operation
            z = soft_thresholding(z + sigma * self.R(xbar), beta)

            # compute second proximal prox_tau_g
            x_km1 = xk.clone()
            
            x_temp = x_km1 - tau * self.RT(z)

            for _ in range(n_inner_iter):
                x_rec = xk - self.AT( weights * (self.A(xk) - b)) / D_rec
                xk = (D_rec * x_rec + x_temp / tau) / ( D_rec + 1 / tau) 
                xk[xk < 0] = 0

            xbar = xk + theta * (xk - x_km1)

            if k % 25 == 0:
                plt.imshow(xbar[0, 0, :, :].cpu(), cmap='gray')
                plt.show()

        return
    
    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.radon.transform(x)
            
    def AT(self, x: torch.Tensor) -> torch.Tensor:
        return self.radon.transposed_transform(x)

    def R(self, x: torch.Tensor, factor: float = -1.0) -> torch.Tensor:
        return self.regularizer.transform(x, factor)
            
    def RT(self, x: torch.Tensor, factor: float = -1.0) -> torch.Tensor:
        return self.regularizer.transposed_transform(x, factor)
