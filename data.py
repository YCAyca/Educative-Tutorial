import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import glob
import cv2
import random
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.utils import make_grid 


class FlowersDataset(Dataset):
    def __init__(self, image_paths, class_idx, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_idx = class_idx
         
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_filepath.split('/')[-2]
        label = self.class_idx[label]
        if self.transform is not None:
            image = self.transform(image)
         
        return image, label


def split_dataset(data_path, generator=None):
    class_names = []
    images = []
    
    for folder_name in os.listdir(data_path):
        d = os.path.join(data_path, folder_name)
        class_names.append(folder_name)
        if os.path.isdir(d):
            for img_name in os.listdir(os.path.join(data_path,folder_name)):
                d = os.path.join(data_path, folder_name, img_name)
                images.append(d)
            
    class_idx = {j : i for i, j in enumerate(class_names)}
  
    dataset_length = len(images)


    train_size = int(dataset_length * 6 / 10) # assign %60 of the images to the train dataset
    test_size = int(dataset_length * 2 / 10) # assign %20 of the images to the test dataset
    val_size = dataset_length - train_size - test_size  # assign %20 of the images to the validation dataset

    
    print("Dataset includes", dataset_length, " images\n", train_size, " of them assigned into train", 
          test_size, " of them assigned into in test", val_size, " of them assigned into validation sub dataset")
    
    train_idx, test_idx, val_idx = random_split(images, [train_size, test_size, val_size], generator=generator) 

    train_list=[images[i] for i in train_idx.indices]
    test_list=[images[i] for i in test_idx.indices]
    val_list=[images[i] for i in val_idx.indices]
    
    return train_list, val_list, test_list, class_idx