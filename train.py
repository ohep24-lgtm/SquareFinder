import torch
import numpy as np
from model import Model

model = Model()
dataset_file = 'dataset.npz'
data = np.load(dataset_file)
X = data['X']
y = data['y']
print("X shape:", X.shape)
print("Class counts:", np.unique(y, return_counts=True))

batch_size = 16
epochs = 30

for epoch in range(epochs):
    indices = np.random.permutation(len(X))  # shuffle every epoch
    total_loss = 0
    num_batches = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        inp = torch.tensor(X[batch_idx], dtype=torch.float32)  # already (B, 1, H, W)
        target = torch.tensor(y[batch_idx], dtype=torch.float32)

        model.optimizer.zero_grad()
        output = model(inp)
        loss = model.criterion(output.squeeze(-1), target)
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    print(f"Epoch {epoch+1}/{epochs} - avg loss: {total_loss/num_batches:.4f}")

model.save('models.pt')













# ps everything bellow is test methods so we know how accurate our thing is dont use these in our proper program as i got from claude 
correct = 0
with torch.no_grad():
    for i in range(len(X)):
        inp = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
        output = model(inp)
        pred = (torch.sigmoid(output) > 0.5).float().item()
        if pred == y[i]:
            correct += 1
accuracy = correct / len(X)
print(f"Training accuracy: {accuracy*100:.2f}%")


correct_by_class = {0: 0, 1: 0}
total_by_class = {0: 0, 1: 0}
with torch.no_grad():
    for i in range(len(X)):
        inp = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
        output = model(inp)
        pred = (torch.sigmoid(output) > 0.5).float().item()
        label = int(y[i])
        total_by_class[label] += 1
        if pred == label:
            correct_by_class[label] += 1
for label in total_by_class:
    acc = correct_by_class[label] / total_by_class[label]
    print(f"Class {label}: {acc*100:.2f}% accuracy ({total_by_class[label]} samples)")

model.save('models.pt')