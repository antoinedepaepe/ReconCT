from src.models.unet_diffusion import UNet
from src.trainers.ddpm_trainer import GaussianDiffusionTrainer

from src.utils.dataloader import ct_train_dataloader
import torch.nn as nn

device = 'cuda'

model = UNet(in_chan=1,
             out_chan=1,
             T=1000,
             ch=128,
             ch_mult=[1, 2, 2, 2],
             attn=[1],
             num_res_blocks=2,
             dropout=0.1)

model.to(device)
model.train()

loss = nn.MSELoss()

gdt = GaussianDiffusionTrainer(train_dataloader=ct_train_dataloader,
                                val_dataloader=None,
                                loss=loss,
                                model=model,
                                epochs=200,
                                device=device)

gdt.train()