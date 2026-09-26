import torch
from torch import nn
from torch.optim import SGD
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
           nn.Linear(256, 3000),# slightly changed the input size to match the test data may have a little tweaking requiered later idk 
            nn.ReLU(),
            nn.Linear(3000, 1)
        )
        self.optimizer = SGD(self.parameters(), lr=0.01)
        self.criterion = nn.BCEWithLogitsLoss() 

      
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def trainn(self, inp, target, verbose=True):
        self.optimizer.zero_grad()
        output = self.forward(inp)
        loss = self.criterion(output.squeeze(), target.float())
        loss.backward()
        self.optimizer.step()
        if verbose:
            print(loss.item())
    def save(self, file_name='Models/Model.pth'):
        torch.save(self.state_dict(), file_name)
    def load(self, file_name='Models/Model.pth'):
        self.load_state_dict(torch.load(file_name))



