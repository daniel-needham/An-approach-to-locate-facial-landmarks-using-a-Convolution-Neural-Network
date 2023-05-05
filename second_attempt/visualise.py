import torch.utils.data
from dataset import FaceAlignmentDataset
from transforms import Transforms
import matplotlib.pyplot as plt
from utility import *
from torch.utils.data import DataLoader as Dataloader
from model2 import ConvNet
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# hyperparameters
epochs = 1
learning_rate = 0.001
momentum = 0.9
batch_size = 1
scale_down = 2
image_dims = int(256 / scale_down)
num_of_points = 44

#setup datasets
dataset = FaceAlignmentDataset('training_images_full.npz',transform=Transforms())
train_dataset_length, test_dataset_length, val_dataset_length = int(len(dataset) * 0.7) + 1, int(len(dataset) * 0.2), int(len(dataset) * 0.1)

assert (train_dataset_length + test_dataset_length + val_dataset_length) == len(dataset), "Dataset lengths do not add up to total dataset length"
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), int(len(train_dataset) * 0.1)])

#setup dataloaders
train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = Dataloader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


#delete this shit
image, image_original, landmarks = next(iter(train_loader))


image, image_original, landmarks = dataset[0]
print(landmarks)

unnorm = unnormalise(image, landmarks)
display(image,unnorm)
