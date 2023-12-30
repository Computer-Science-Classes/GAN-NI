# src/generator/generator
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)
