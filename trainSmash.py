import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_dataset():
    data_path = 'trainingSet/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform = torchvision.transforms.Compose([
                             transforms.Scale(128),
                             transforms.CenterCrop(128),
                             transforms.ToTensor(),
                             normalize,
                         ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def load_valdataset():
    data_path = 'validationSet/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform = torchvision.transforms.Compose([
                             transforms.Scale(128),
                             transforms.CenterCrop(128),
                             transforms.ToTensor(),
                             normalize,
                         ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return test_loader

trainloader = load_dataset()
testloader = load_valdataset()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)

# print(labels)
# show images
# imshow(torchvision.utils.make_grid(images))

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
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                print(running_loss, (100 * correct / total))
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

torch.save(net.state_dict(), 'convNet')

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        print(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
