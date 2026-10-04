import torch
from torch import nn
from torch.optim import SGD

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.AdaptiveMaxPool2d(1), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),              # (B, 8, 1, 1) -> (B, 8) no more positional bias like previous attempts 
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def trainn(self, inp, target, verbose=True):
        self.optimizer.zero_grad()
        output = self.forward(inp)
        loss = self.criterion(output.squeeze(-1), target.float())
        loss.backward()
        self.optimizer.step()
        if verbose:
            print(loss.item())
        return loss.item()

    def save(self, file_name='models/model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='models/model.pth'):
        self.load_state_dict(torch.load(file_name, weights_only=True))