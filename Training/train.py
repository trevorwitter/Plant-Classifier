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

from utils import download_dataset, PlantImageDataset, LabelTransform, accuracy_score
from model import Net

def training_loop(net, trainloader, valloader, gpu=False, epochs=1):
    if gpu == False:
        device = torch.device("cpu")
    elif gpu == True:
        device = torch.device("mps")
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            train_loss = criterion(outputs.squeeze(1), labels)
            train_loss.backward()
            optimizer.step()
            
            # print statistics
            print(f"{train_loss.item()}")
            running_loss += train_loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.8f}')
                running_loss = 0.0
        train_acc = accuracy_score(net, trainloader, gpu=gpu)
        val_acc = accuracy_score(net, valloader, gpu=gpu)
        print(f"Epoch {epoch} train_acc: {train_acc}, val_acc: {val_acc}")
    PATH = './models/plant_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Training complete - model saved to {PATH}')


def main(epochs=1, gpu=False, num_workers=8):
    #download_dataset()

    transform = transforms.Compose(
        [transforms.Resize(256,antialias=True),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    label_transform = LabelTransform()
    train_data = PlantImageDataset(annotations_file="data/annotations_train_file.csv",
                             img_dir="data/2021_train_mini/",
                             transform=transform,
                             target_transform=label_transform.transform
                             )

    val_data = PlantImageDataset(annotations_file="data/annotations_val_file.csv",
                             img_dir="data/2021_train_mini/",
                             transform=transform,
                             target_transform=label_transform.transform
                             )

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=num_workers)

    X, y = next(iter(train_dataloader))
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    net = Net()
    training_loop(net, train_dataloader, val_dataloader, gpu=gpu, epochs=epochs)

if __name__ == "__main__":
    epochs=1
    gpu=True
    num_workers=8
    
    main(
        epochs=epochs,
        gpu=gpu, 
        num_workers=num_workers,
        )
