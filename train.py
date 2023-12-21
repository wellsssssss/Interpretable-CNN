import torch, torchvision
from torch import nn, optim
from dataloader import train_loader,test_loader
from torch.autograd import Variable
import numpy as np
import shap
import pickle
from model import CNNModel
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
batch_size = 128
num_epochs = 100
# Create CNN
model = CNNModel()
model.to(device)
# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 1, 176, 208))
        labels = Variable(labels)

        optimizer.zero_grad()  # Clear gradients
        outputs = model(train)  # Forward propagation
        loss = error(outputs, labels)  # Calculate softmax and cross entropy loss
        loss.backward()  # Calculating gradients
        optimizer.step()  # Update parameters

        count += 1

        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0

            # Predict test dataset
            for images, labels in test_loader:
                test = Variable(images.view(-1, 1, 176, 208))
                outputs = model(test)  # Forward propagation
                predicted = torch.max(outputs.data, 1)[1]  # Get predictions from the maximum value
                total += len(labels)  # Total number of labels
                correct += (predicted == labels).sum()  # Total correct predictions

            accuracy = 100.0 * correct.item() / total

            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 50 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data.item(), accuracy))

batch = next(iter(test_loader))
images, _ = batch
# images = images.view(-1, 1, 176, 208)
images = images.view(-1, 1, 208,176)
background = images[:100]
test_images= images[100:103]

e = shap.DeepExplainer(model, images)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy)
# 假设您的模型对象为model
model_state_dict = model.state_dict()

# 保存state_dict到文件
with open('D:/model_params.pkl', 'wb') as f:
    pickle.dump(model_state_dict, f)