import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as func
import numpy as np
from torchvision.datasets import FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler

# Constants.
CONST_INPUT_SIZE = 28*28
CONST_LEARNING_RATE = 0.2  # 0.2 - 87.84 (AdaDelta)
CONST_EPOCH_NUMBER = 10
CONST_BATCH_SIZE = 64
CONST_NONE = 0


# The FirstNet class.
class FirstNet(nn.Module):
    def __init__(self, opt_model, input_size):
        super(FirstNet, self).__init__()
        self.opt_model = opt_model
        self.input_size = input_size
        self.fc_0 = nn.Linear(input_size, 100, bias=True)
        self.fc_1 = nn.Linear(100, 50, bias=True)
        self.fc_2 = nn.Linear(50, 10, bias=True)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    # The ff method.
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.bn1(func.relu(self.fc_0(x)))
        x = func.dropout(x, 0.5, True, True)
        x = self.bn2(func.relu(self.fc_1(x)))
        x = func.dropout(x, 0.5, True, True)
        x = self.fc_2(x)
        return func.log_softmax(x, CONST_NONE)


# Train and test.
def train_and_validate(model, optimizer):
    for epoch_number in range(1, CONST_EPOCH_NUMBER + 1):
        train(epoch_number, model, optimizer)
        test(model)


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
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_loader)
    print('== TRAIN == Epoch number: {}, Average loss: {:.4f}'.format(epoch_number, avg_loss))


# The test method.
def test(model):
    model.eval()
    test_loss = 0
    correct = 0.0
    for data, target in validation_loader:
        output = model(data)
        test_loss += func.nll_loss(output, target, size_average=False).data  # sum up batch loss
        prediction = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    test_loss /= len(validation_loader.sampler.indices)
    print('== TEST == Model name: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:03.2f}%)'.format(
        model.opt_model, test_loss, correct, len(validation_loader.sampler.indices), float(100.0 * correct) / len(validation_loader.sampler.indices)))


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

# Create the model.
SGDModel = FirstNet("SGD", input_size=CONST_INPUT_SIZE)
SGDOptimizer = optim.SGD(SGDModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(SGDModel, SGDOptimizer)

AdamModel = FirstNet("Adam", input_size=CONST_INPUT_SIZE)
AdamOptimizer = optim.Adam(AdamModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(AdamModel, AdamOptimizer)

AdaDeltaModel = FirstNet("AdaDelta", input_size=CONST_INPUT_SIZE)
AdaDeltaOptimizer = optim.Adadelta(AdaDeltaModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(AdaDeltaModel, AdaDeltaOptimizer)

RMSPropModel = FirstNet("RMSProp", input_size=CONST_INPUT_SIZE)
RMSPropOptimizer = optim.RMSprop(RMSPropModel.parameters(), lr=CONST_LEARNING_RATE)
train_and_validate(RMSPropModel, RMSPropOptimizer)