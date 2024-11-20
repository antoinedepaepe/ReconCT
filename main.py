from src.operators.radon import Radon
from src.operators.total_variation import TotalVariation
from src.solvers.pwls import PWLS


from PIL import Image
import torch
import numpy as np
import torch_radon as tr

device = 'cuda'

n_rays = 128
height, width = 128, 128
n_angles = 360
volume = tr.Volume2D()
volume.set_size(height=height, width=width)
angles = torch.linspace(0, 2 * torch.pi * ((n_angles - 1))/n_angles, n_angles, device=device)

radon = Radon(n_rays=n_rays,
              angles=angles, 
              volume=volume)

tv = TotalVariation(penalty='l2')

I = 3e5
EPS = 1e-10


img_path = '/home/adepaepe/Data/dev/DeepMove/data/motion/6.png'
x_true = np.array(Image.open(img_path).convert('L'))
x_true = torch.from_numpy(x_true).float().to(device)


sino = radon.transform(x_true)
yi = torch.poisson(I*torch.exp(-sino))
noisy_sino = torch.log(I/ (yi + EPS))


x0 = torch.zeros_like(x_true, device=device)
b = noisy_sino
beta = 0.1
n_iter = 200
weights = yi


solver = PWLS(radon=radon, regularizer=tv)
solver.solve(x0,
             b,
             beta,
             n_iter,
             weights)