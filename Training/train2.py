import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image

from utils import download_dataset, PlantImageDataset, label_transform
from model import Net

def training_loop(net, trainloader, valloader, gpu=False, epochs=1):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        device = torch.device("mps")
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs): # loop over thef dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            train_loss = criterion(outputs, labels)

            #val_outputs = net(val_inputs)
            #val_loss = criterion(val_outputs, val_labels)
            train_loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += train_loss.item()
            #print(train_loss.item())
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.8f}')
                running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)
                #print(val_loss.item())

    print('Training Complete')


def main():
    download_dataset()

    transform = transforms.Compose(
        [transforms.Resize(256,antialias=True),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        #transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_data = PlantImageDataset(annotations_file="data/annotations_train_file.csv",
                             img_dir="data/2021_train_mini/",
                             transform=transform,
                             target_transform=label_transform)

    val_data = PlantImageDataset(annotations_file="data/annotations_val_file.csv",
                             img_dir="data/2021_train_mini/",
                             transform=transform,
                             target_transform=label_transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=True, num_workers=8)

    X, y = next(iter(train_dataloader))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    net = Net()
    training_loop(net, train_dataloader, val_dataloader, gpu=True)

if __name__ == "__main__":
    main()
