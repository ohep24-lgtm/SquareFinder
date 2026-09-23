import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),           
            nn.Linear(26244, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
