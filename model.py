import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32 * 32 * 64, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
