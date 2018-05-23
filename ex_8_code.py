import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

# Constants.
CONST_SIZE = 28*28


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc_0 = nn.Linear(image_size, 100)
        self.fc_1 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        return F.log_softmax(x)


transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(FashionMNIST('./data', train=True, download=True, transform=transforms),
                                           batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(FashionMNIST('./data', train=False, transform=transforms),
                                          batch_size=64, shuffle=True)

model = FirstNet(image_size=CONST_SIZE)

print("Hello World!")