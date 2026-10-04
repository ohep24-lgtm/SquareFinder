import torch
import numpy as np
from model import Model

model = Model()
dataset_file = 'dataset.npz'
data = np.load(dataset_file)
X = data['X']
y = data['y']
print("X shape:", X.shape)
print("Class counts:", np.unique(y, return_counts=True)) #prints out the number of samples in each class
'''
++++++++++++++train/validation split happens here++++++++++++++
'''
np.random.seed(0)   # seeds with a fixed nr, for reproducable results on training/validation 
n = len(X)
perm = np.random.permutation(n)
val_size = max(1, int(0.15 * n))         # ~15% held out for validation, with 1 sample as a safeguard for small datasets 
val_idx, train_idx = perm[:val_size], perm[val_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]   # doesn't update the model's weights, so it can be used to evaluate the model's performance on unseen data during training
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

batch_size = 16     #how many images get bundled together per weight update, instead of updating after every image 
epochs = 300    #how many full passes over the training data

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))   # shuffle every epoch (of the training indices)
    total_loss = 0  #reset at the start of every epoch
    num_batches = 0 #reset at the start of every epoch 

#++++++++++++++++++++++++++++++++++++++++++++++++++ everything between this line 
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        inp = torch.tensor(X_train[batch_idx], dtype=torch.float32)
        target = torch.tensor(y_train[batch_idx], dtype=torch.float32)

        model.optimizer.zero_grad()
        output = model(inp)
        loss = model.criterion(output.squeeze(-1), target)
        loss.backward()
        model.optimizer.step()
#++++++++++++++++++++++++++++++++++++++++++++++++++ and between this line respects the standard training steps

        total_loss += loss.item()   # pulls the plain Python number from the tensor, so we can keep track of the loss over the entire epoch
        num_batches += 1    

    model.eval() # puts the model in evaluation mode

#++++++++++++++++++++++++++++++++++++++++++++++++++ everything between this line 
    with torch.no_grad():
        val_inp = torch.tensor(X_val, dtype=torch.float32)
        val_target = torch.tensor(y_val, dtype=torch.float32)
        val_output = model(val_inp)
        val_loss = model.criterion(val_output.squeeze(-1), val_target).item()
#++++++++++++++++++++++++++++++++++++++++++++++++++ and between this line runs the entire validation set through the model (in one forward pass) without updating weights 
    
    model.train() # puts the model back in training mode

    print(f"Epoch {epoch+1}/{epochs} - train loss: {total_loss/num_batches:.4f}  val loss: {val_loss:.4f}")

model.save('models.pt')
'''
Reusable function that evaluates the model on a given dataset and prints the accuracy for each class and overall accuracy.
'''
def evaluate(X_eval, y_eval, label=""):
    model.eval()
    correct = 0
    correct_by_class = {0: 0, 1: 0}
    total_by_class = {0: 0, 1: 0}
    with torch.no_grad():
        #loops trough the given dataset one image at a time, makes a prediction, and compares it to the true label to calculate accuracy
        for i in range(len(X_eval)):
            inp = torch.tensor(X_eval[i], dtype=torch.float32).unsqueeze(0)
            output = model(inp)
            pred = (torch.sigmoid(output) > 0.5).float().item()
            true_label = int(y_eval[i])
            total_by_class[true_label] += 1
            if pred == true_label:
                correct += 1
                correct_by_class[true_label] += 1

    #prints overall accuracy, then loops over each class to print individual accuracy (while making sure its not dividing by zero if class happens to have zero samples)
    acc = correct / len(X_eval)
    print(f"\n{label} accuracy: {acc*100:.2f}%  ({len(X_eval)} samples)")
    for cls in total_by_class:
        if total_by_class[cls] > 0:
            cls_acc = correct_by_class[cls] / total_by_class[cls]
            print(f"  class {cls}: {cls_acc*100:.2f}%  ({total_by_class[cls]} samples)")

#final print 
evaluate(X_train, y_train, "Training")
evaluate(X_val, y_val, "Validation")