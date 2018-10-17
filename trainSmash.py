import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F



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


testloader = load_valdataset()

# get some random training images


# print(labels)
# show images
# imshow(torchvision.utils.make_grid(images))

trainloader = load_dataset()
dataiter = iter(trainloader)
images, labels = dataiter.next()

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

                print(epoch ,running_loss, (100 * correct / total))
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
