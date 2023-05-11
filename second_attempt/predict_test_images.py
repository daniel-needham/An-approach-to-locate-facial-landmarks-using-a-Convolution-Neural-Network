import numpy
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
dataset = FaceAlignmentDataset('test_images.npz',transform=Transforms())
dataloader = Dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# display first image
# dataloader = iter(dataloader)
# image, image_original, points = next(dataloader)
# plt.imshow(image.numpy().squeeze(), cmap='gray')
# plt.imshow(image_original.numpy().squeeze())
# print(points.numpy().squeeze())

# print dataset length
print(len(dataset))

# create results matrix
results = numpy.zeros((len(dataset), 44, 2))
print(results.shape)

# load model
model = ConvNet(128, 44)
model.load_state_dict(torch.load('face_landmarks_extended_dataset_1.pth')) # best model
model.eval()

# display predictions
with torch.no_grad():
    for batch_i, batch in enumerate(dataloader):
        image, image_original, _ = batch
        y_hat = model(image)
        y_hat = y_hat.view(y_hat.size(0), -1, 2)
        y_hat_unnorm = unnormalise(image_original[0], y_hat[0])
        results[batch_i] = y_hat_unnorm

        if batch_i  % 99 == 553:
            display_from_normalised(image[0], y_hat[0])
            display_from_normalised(image_original[0], y_hat[0])



# save results
save_as_csv(results)
