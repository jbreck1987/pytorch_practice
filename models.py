"""
Model definitions to be used in experiments
"""

# Author: Josh Breckenridge
# Data: 7-21-2023

from torch import nn


class ConvRegv1(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=60, stride=1, padding='same'), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same'), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding='same'), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.block4 = nn.Sequential(
            nn.Linear(in_features=32*125, out_features=100),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(in_features=100, out_features=1)
        )
    
    def forward(self, x):
        return self.block4(self.block3(self.block2(self.block1(x))))