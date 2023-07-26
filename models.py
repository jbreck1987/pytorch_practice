"""
Model definitions to be used in experiments
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

import torch


class ConvRegv1(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=200, stride=1, padding='same'), 
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=60, stride=1, padding='same'), 
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=15, stride=1, padding='same'), 
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Flatten()
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*125, out_features=100),
            torch.nn.ReLU(),
            #torch.nn.Dropout1d(0.1),
            torch.nn.Linear(in_features=100, out_features=1)
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x)))).unsqueeze(dim=1)