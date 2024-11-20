import torch
import torch_radon as tr
from src.operators.operator import Operator

class Radon(Operator):

    def __init__(self, n_rays: int, 
                       angles: torch.Tensor,
                       volume: tr.Volume2D) -> None:
        
        self.radon = tr.ParallelBeam(det_count = n_rays, 
                                     angles = angles, 
                                     volume = volume)

    def transform(self, x: torch.tensor) -> torch.tensor:
        return self.radon.forward(x)

    def transposed_transform(self, x: torch.tensor) -> torch.tensor:
        return self.radon.backward(x)

    def fbp(self, x: torch.tensor) -> torch.tensor:
        sinogram = self.radon.forward(x)
        filtered_sinogram = self.radon.filter_sinogram(sinogram)
        fbp = self.radon.backward(filtered_sinogram)
        return fbp

