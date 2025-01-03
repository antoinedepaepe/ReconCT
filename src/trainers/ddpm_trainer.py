import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from src.trainers.trainer import Trainer
from typing import Tuple


# TODO : move this and maybe change it
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(Trainer): 
    
    def __init__(self,  train_dataloader: DataLoader,
                        val_dataloader: DataLoader,
                        model: nn.Module,
                        loss: nn.Module,
                        learning_rate: float = 1e-4,
                        epochs: int = 100,
                        grad_clip: float = 1e-1,
                        weights_dir: str = './weights',
                        beta1: float = 1e-4, 
                        betaT: float = 0.02,
                        T: int = 1000,
                        device: str = 'cuda') :
        
        super().__init__(train_dataloader=train_dataloader,
                         val_dataloader= DataLoader,
                         model=model,
                         loss=loss,
                         learning_rate=learning_rate,
                         epochs=epochs,
                         grad_clip=grad_clip,
                         weights_dir=weights_dir)
    
        self.device = device
        
        self.T = T
        self.betas = torch.linspace(beta1, betaT, T).double().to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar).to(device)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar).to(device)

    def one_step(self, packed: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        
        x0 = packed.to(self.device)

        b, c, h, w = x0.shape

        t = torch.randint(self.T, size=(b, )).to(self.device)
        epsilon = torch.randn_like(x0).to(self.device)
        xt = (
            extract(self.sqrt_alphas_bar, t, (b, c, h, w)) * x0 +
            extract(self.sqrt_one_minus_alphas_bar, t, (b, c, h, w)) * epsilon )
                
        x, y = self.model(xt, t), epsilon

        return x, y



if __name__ == "__main__":

    x0 = torch.rand((16, 3, 32, 32)).to('cuda')

    model = nn.Identity()
    beta1 = 1.0
    betaT = 10.0

    diffusion = GaussianDiffusionTrainer(model, beta1, betaT)

    res = diffusion.train(x0)