import torch
import numpy as np
import torch
from Model import Model

model = Model()
dataset_file = 'dataset.npz'

data = np.load(dataset_file)
X = data['X']   
y = data['y'] 

for _ in range(1):
    for i in range(len(X)):
        inp = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)   
        target = torch.tensor(float(y[i]))                          
        print(target)
        model.trainn(inp, target)

model.save('square_model.pt')
# why dont we use batching ?