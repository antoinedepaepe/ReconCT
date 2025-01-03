import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

class Trainer(ABC):

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: nn.Module,
        loss: nn.Module,
        learning_rate: float = 1e-4,
        epochs: int = 100,
        grad_clip: float = 1e-1,
        weights_dir: str = './weights'):

        super().__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.model = model
        self.loss = loss
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.grad_clipper = lambda model_weights: torch.nn.utils.clip_grad_norm_(model_weights, grad_clip)

        self.weights_dir = weights_dir
        
    def save_model_weights(self, epoch: int) -> None:
        
        model_path = os.path.join(self.weights_dir, f"wavelet128x128x128_standardized_diverse_model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        print("Model weights saved to {model_path}")

    def train(self) -> None:
        
        nbatchs = len(self.train_dataloader)
        n_iter = 0

        for epoch in range(self.epochs):
            cum_loss = 0.0
            for packed in tqdm(self.train_dataloader):
                
                x, y = self.one_step(packed)

                loss = self.loss(x, y)
                cum_loss += loss.item()

                if n_iter % 25 == 0:
                    print(f'Iter : {n_iter}, loss : {loss.item()}')

                # otpimization steps
                loss.backward()

                del loss
                torch.cuda.empty_cache()
                
                self.grad_clipper(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                n_iter += 1
            
            cum_loss /= nbatchs
            
            print(f"Epoch {epoch}, loss on train : {cum_loss}")

            self.save_model_weights(epoch)
        
        # TODO: add a loop on validation set to evaluate results
            
    @abstractmethod
    def one_step(self):
        pass