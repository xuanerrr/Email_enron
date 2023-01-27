import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes as nb
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
from data_process import get_features

max_features = 5000
class_num = 2


x,y = get_features()

X_train, X_test, y_train, y_test = train_test_split(x, np.asarray(y), test_size=0.2, random_state=11)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#(25321, 5000) (6331, 5000) (25321,) (6331,)


device = "cuda" if torch.cuda.is_available() else "cpu"

# 实例化Dataset
class EmailDataset(Dataset):
    def __init__(self, data, label):
        self.labels = label
        self.emaildata = data
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emaildata_idx = self.emaildata[idx]
        label_idx = self.labels[idx]
        return emaildata_idx, label_idx

train_data = EmailDataset(X_train, y_train)
test_data = EmailDataset(X_test, y_test)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

print(len(train_dataloader.dataset))
# for batch, (X, y) in enumerate(train_dataloader):
#     print(X.shape)

# 建立神经网络
class Net(torch.nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(max_features, 128)
        self.out = torch.nn.Linear(128, class_num)
        

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x)
        return x

net = Net().to(device)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

def train(dataloader, model, optimizer, loss_func):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.to(torch.float32))
        loss = loss_func(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 200 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"[{current:>5d}/{size:>5d}]  loss: {loss:>7f}")
    print(f"Train loss = {loss:>7f}")

def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.to(torch.float32))
            test_loss += loss_func(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test: accuracy = {(100*correct):>0.1f}%,  loss = {test_loss:>8f}")

epochs = 50
for t in range(epochs):
    print(f"------Epoch {t+1}------")
    train(train_dataloader, net, optimizer, loss_func)
    test(test_dataloader, net, loss_func)
