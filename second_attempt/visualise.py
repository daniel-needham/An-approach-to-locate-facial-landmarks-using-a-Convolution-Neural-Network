import torch.utils.data
from dataset import FaceAlignmentDataset
from transforms_predict import Transforms
import matplotlib.pyplot as plt
from utility import *
from torch.utils.data import DataLoader as Dataloader
from numpy import arange
from model import ConvNet
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

batch_size = 1

# setup dataset and dataloader
dataset = FaceAlignmentDataset('examples.npz',transform=Transforms())
dataloader = Dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataloader = iter(dataloader)
image = next(dataloader)[0]

plt.imshow(image.numpy().squeeze(), cmap='gray')
plt.show()

# load model
model = ConvNet(128, 44)
model.load_state_dict(torch.load('face_landmarks_extended_dataset_1.pth'))
model.eval()

# display predictions
with torch.no_grad():
    for batch_i, batch in enumerate(dataloader):
        image, image_original, _ = batch
        y_hat = model(image)
        y_hat = y_hat.view(y_hat.size(0), -1, 2)
        display_from_normalised(image[0], y_hat[0])
        display_from_normalised(image_original[0], y_hat[0])

