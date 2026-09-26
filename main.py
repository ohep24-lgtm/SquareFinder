import torch
from create_dataset import ConvertDir
from model import Model          # match your actual filename's case
from draw import Canvas

c = Canvas(save_dir='cache')
c.run()                         

m = Model()
m.load('models.pt')  # load the trained model
m.eval()

conv = ConvertDir("cache", "cache")
inp = torch.tensor(conv.image_arrays[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    output = m(inp)

prob_square = torch.sigmoid(output[0]).item()
print(f"square: {prob_square*100:.2f}%")
print(f"not square: {(1-prob_square)*100:.2f}%")
