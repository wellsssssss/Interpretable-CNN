import torch, torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
batch_size = 128
device=torch.device('cpu')
train = pd.read_csv("D:/archive/combined.csv",dtype = np.float32)
target = train.label.values
train = train.loc[:,train.columns != "label"].values/255

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 42)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long
X_train, X_test, y_train, y_test=X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
# num_epochs = 50

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)