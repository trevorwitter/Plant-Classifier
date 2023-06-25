import os
import pandas as pd
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F


def download_dataset():
    if os.path.isdir('./data/2021_train_mini/') == False:
        trainset = torchvision.datasets.INaturalist(root='./data/',
                                                    version='2021_train_mini',
                                                    target_type='full',
                                                    download='True',
                                                    )
        plants = ['data/2021_train_mini/'+ x + "/" for x in os.listdir('data/2021_train_mini/') 
                if x.split('_')[1] == 'Plantae']
        not_plants = ['data/2021_train_mini/' + x + "/" for x in os.listdir('data/2021_train_mini/') 
                    if x.split('_')[1] != 'Plantae']
        for x in not_plants:
            shutil.rmtree(x)

    #labels = set([x.split('_')[2] for x in os.listdir('data/2021_train_mini') 
    #            if x.split('_')[1] == 'Plantae'])

    #label_map = {x:y for x,y in zip(labels, range(len(labels)))}
    img_locs = []
    for path, subdirs, files in os.walk('data/2021_train_mini'):
        for file in files:
            if file.startswith("."):
                pass
            else:
                img = path[21:] + "/" + file
                img_locs.append(img)
    subset = list(set([x.split("Plantae_")[1].split("/")[0] for x in img_locs]))[:14]
    img_locs = [x for x in img_locs if x.split("Plantae_")[1].split("/")[0] in subset]
    labels = [x.split("Plantae_")[1].split("_")[1] for x in img_locs]
    '''labels_map = {
        "Bryophyta":0,
        "Chlorophyta":1,
        "Marchantiophyta":2,
        "Rhodophyta":3,
        "Tracheophyta":4,
    }'''
    labels_map = {}
    i = 0
    for x in set(labels):
        labels_map[x] = i
        i += 1
    print(f"Number of classes: {len(labels_map)}")
    
    annotations = pd.DataFrame(zip(img_locs, [labels_map[x] for x in labels]),columns=['img_loc','label'])
    #Shuffle annotations df
    annotations = annotations.sample(frac=1).reset_index(drop=True)
    val_idx = int(len(annotations)*.8)
    test_idx = int(len(annotations)*.9)
    annotations_train = annotations.iloc[:val_idx]
    annotations_val = annotations.iloc[val_idx:test_idx]
    annotations_test = annotations.iloc[test_idx:]
    annotations_train.to_csv("data/annotations_train_file.csv",index=False)
    annotations_val.to_csv("data/annotations_val_file.csv",index=False)
    annotations_test.to_csv("data/annotations_test_file.csv",index=False)
    #else:
        #print("data already downloaded")

def label_transform(labels, num_classes=14):
    labels_tensor = torch.tensor(labels)
    one_hot = F.one_hot(labels_tensor, num_classes=num_classes)
    return one_hot.type(torch.float32)

class PlantImageDataset(torch.utils.data.Dataset):
    """For working with plant images from the inaturalist dataset"""
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        label = self.img_labels.iloc[idx, 1:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc