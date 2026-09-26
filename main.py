import torch
from create_dataset import ConvertDir
from Model import Model          # match your actual filename's case
from draw import Canvas

c = Canvas(save_dir='cache')
c.run()                          # opens the window; blocks until you quit/close it

m = Model()
m.load()
m.eval()

conv = ConvertDir("cache", "cache")
inp = torch.tensor(conv.image_arrays[0], dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output = m(inp)

print("square:", output[0].item() * 100)
print("not square:", output[1].item() * 100)