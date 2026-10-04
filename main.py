import torch
from PIL import Image
import numpy as np
from model import Model
from draw import Canvas
from create_dataset import SIZE

c = Canvas(save_dir='cache')
c.run()

if c.last_saved is None:
    raise RuntimeError("No drawing was saved — press 's' before closing the window.")

m = Model()
m.load('models.pt')
m.eval()

im = Image.open(c.last_saved).convert('L').resize(SIZE)
arr = 1.0 - (np.array(im, dtype=np.float32) / 255.0)   

inp = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

with torch.no_grad():
    output = m(inp)

prob_not_square = torch.sigmoid(output[0]).item()
prob_square = 1 - prob_not_square
print(f"square: {prob_square*100:.2f}%")
print(f"not square: {(prob_not_square)*100:.2f}%")
