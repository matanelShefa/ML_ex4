import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as func
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler


# Constants.
CONST_INPUT_SIZE = 28*28
CONST_LEARNING_RATE = 0.2  # 0.2 - SGD - 89.35, 88.58, 87.25
CONST_EPOCH_NUMBER = 10
CONST_BATCH_SIZE = 64
CONST_NONE = 0


# The FirstNet class.
class FirstNet(nn.Module):
    def __init__(self, opt_model):
        super(FirstNet, self).__init__()
        self.opt_model = opt_model
        self.input_size = CONST_INPUT_SIZE
        self.fc_0 = nn.Linear(CONST_INPUT_SIZE, 100, bias=True)
        self.fc_1 = nn.Linear(100, 50, bias=True)
        self.fc_2 = nn.Linear(50, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    # The ff method.
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = func.relu(self.fc_0(x))
        #x = func.relu(self.bn1(self.fc_0(x)))  # Batch normalization
        #x = func.dropout(x, 0.5, True, True)  # Dropout
        x = func.relu(self.fc_1(x))
        #x = func.relu(self.bn2(self.fc_1(x)))  # Batch normalization
        #x = func.dropout(x, 0.5, True, True)  # Dropout
        x = self.fc_2(x)
        return func.log_softmax(x, CONST_NONE)


# Train and test.
def train_and_validate(model, optimizer, train_loss, validation_loss):
    for epoch_number in range(1, CONST_EPOCH_NUMBER + 1):
        train_loss.append(train(epoch_number, model, optimizer))
        validation_loss.append(test(model, validation_loader))


# The training method.
def train(epoch_number, model, optimizer):
    model.train()
    avg_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = func.nll_loss(output, labels)
        avg_loss += loss
        prediction = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        prediction_list.append(prediction)
        correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_loader)
    print('== TRAIN == Epoch number: {}, Average loss: {:.4f} , Accuracy: {}/{} ({:03.2f}%)'.format(
        epoch_number, avg_loss, correct, len(train_loader.sampler), float(100.0 * correct) / len(train_loader.sampler)))
    return avg_loss.item()


# The test method.
def test(model, data_set):
    model.eval()
    test_loss = 0
    correct = 0.0
    for data, target in data_set:
        output = model(data)
        test_loss += func.nll_loss(output, target, size_average=False).data  # sum up batch loss
        prediction = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    test_loss /= len(data_set.sampler)
    print('== TEST == Model name: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:03.2f}%)'.format(
        model.opt_model, test_loss, correct, len(data_set.sampler), float(100.0 * correct) / len(data_set.sampler)))
    return test_loss.item()


# Define our FashionMNIST Data sets (Images and Labels) for training and testing.
train_set = FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_set = FashionMNIST(root='./data',train=False, transform=transforms.ToTensor())

# Define the indices
indices = list(range(len(train_set)))  # start with all the indices in training set
split = int(len(train_set) * 0.2)  # define the split size

# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

# Define our samplers.
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

# Create the loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=CONST_BATCH_SIZE, sampler=validation_sampler)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=CONST_BATCH_SIZE, shuffle=False)

# Prediction to print to the file.
prediction_list = []

# Plot the results.
trainLoss = []
validationLoss = []
epochList = np.arange(1, CONST_EPOCH_NUMBER + 1)

# Create the model.
SGDModel = FirstNet("SGD")
SGDOptimizer = optim.SGD(SGDModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(SGDModel, SGDOptimizer, trainLoss, validationLoss)
test(SGDModel, test_loader)

#AdamModel = FirstNet("Adam")
#AdamOptimizer = optim.Adam(AdamModel.parameters(), lr=CONST_LEARNING_RATE)
#train_and_validate(AdamModel, AdamOptimizer, trainLoss, validationLoss)
#test(AdamModel, test_loader)

AdaDeltaModel = FirstNet("AdaDelta")
AdaDeltaOptimizer = optim.Adadelta(AdaDeltaModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(AdaDeltaModel, AdaDeltaOptimizer, trainLoss, validationLoss)
test(AdaDeltaModel, test_loader)

#RMSPropModel = FirstNet("RMSProp")
#RMSPropOptimizer = optim.RMSprop(RMSPropModel.parameters(), lr=CONST_LEARNING_RATE)
#train_and_validate(RMSPropModel, RMSPropOptimizer, trainLoss, validationLoss)
#test(RMSPropModel, test_loader)

with open("test.pred", "w+") as pred:
    pred.write('\n'.join(str(v) for v in prediction_list))

# Set the graph.
fig, ax = plt.subplots()
plt.plot(epochList, trainLoss, 'r', color="red", label="Train Loss")
plt.plot(epochList, validationLoss, 'r', color="blue", label="Validation Loss")

# Set the legend.
legend = ax.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
for label in legend.get_texts():
    label.set_fontsize('large')  # the legend text size
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.ylabel('Average Loss')
plt.xlabel('Epoch')

# Draw the graph.
plt.show()