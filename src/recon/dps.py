import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DeepPosteriorSampling:
    
    """ Basic Vanilla DDPM : Made that way (not like algo2 in the paper)
        because we will need that parametrization when we will use KL div"""

    def __init__(self, 
                 model: nn.Module, 
                 beta1: float = 1e-4, 
                 betaT: float = 0.02,
                 T: int = 1000,
                 device: str = 'cuda'):
       
        super().__init__()

        self.model = model
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta1, betaT, T).double().to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T].to(device)

        self.sqrt_recip_alphas_bar = torch.sqrt(1. /self.alphas_bar).to(device)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1).to(device)

        self.posterior_var = (self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)).to(device)
        self.posterior_log_var_clipped = torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])).to(device)
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)).to(device)
        self.posterior_mean_coef2 = (torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)).to(device)

        # ensure models is turned to eval mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def q_mean_variance(self, x0: torch.Tensor, 
                              xt: torch.Tensor,
                              t: int):

        # mean + log-variance of q(x_{t-1} | xt, x0)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, xt.shape) * x0 +
            extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, xt.shape)
        
        return posterior_mean, posterior_log_var_clipped

    def predict_x0_from_epsilon_theta(self, xt: torch.Tensor, 
                                      t: int,
                                      epsilon_theta: torch.Tensor):
        
        # reparam tricks from q(xt | x0) closed form
        x0 = (
            extract(self.sqrt_recip_alphas_bar, t, xt.shape) * xt -
            extract(self.sqrt_recipm1_alphas_bar, t, xt.shape) * epsilon_theta
        )
        
        return x0


    def p_mean_variance(self, xt: torch.Tensor, 
                              t: int):
        """
        mean + log-variance of p_{theta}(x_{t-1} | xt)
        """
        
        # log param (not necessary not but will be in future when we will learn sigmas)
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, xt.shape)

        # mean param
        epsilon_theta = self.model(xt, t)

        # x0 approx x0_hat(xt, t)
        x0_hat = self.predict_x0_from_epsilon_theta(xt, t, epsilon_theta=epsilon_theta)
        model_mean, _ = self.q_mean_variance(x0_hat, xt, t)
       
        return x0_hat, model_mean, model_log_var

    def inference(self, xT: torch.Tensor,
                        y: torch.Tensor,
                        weights: torch.Tensor,
                        lam: float):
                    
        xt = xT
        
        for time_step in tqdm(reversed(range(self.T))):
            
            xt = xt.detach()
            with torch.no_grad():
                
                t = xt.new_ones([xT.shape[0], ], dtype=torch.long) * time_step
                epsilon = torch.randn_like(xt) if time_step > 0 else 0
                _ , mean, log_var = self.p_mean_variance(xt=xt, t=t)
    
                # torch.exp(0.5 * log_var) -> var ** 0.5
                xt_prime = mean + torch.exp(0.5 * log_var) * epsilon

            with torch.enables_grad():
                xt = xt.requires_grad_()

                # zero existing gradient
                if xt.grad is not None:
                    xt.grad.zero_()

                x0_hat, _, _ = self.p_mean_variance(xt=xt, t=t)
                
                loss = self.data_fidelity(x0_hat, y, weights)
                grad_data_fidelity = xt.grad

                # gradient normalization
                grad_data_fidelity_norm = torch.linalg.norm(grad_data_fidelity)
                grad_data_fidelity = grad_data_fidelity/(grad_data_fidelity_norm+1e-7)

                #gradient descent
                xt = xt_prime - lam * grad
            
        return xt
    

if __name__ == "__main__":
    
    # Define the model
    class Dummy(nn.Module):
        def __init__(self):
            super(Dummy, self).__init__()
            self.conv = nn.Conv2d(3, 3, 1)
        def forward(self, x, t):
            return self.conv(x)

    model = Dummy().to('cuda')
    gs = GaussianDiffusionSampler(model=model)

    x = torch.randn(16, 3, 16, 16).to('cuda')
    gs.inference(x)