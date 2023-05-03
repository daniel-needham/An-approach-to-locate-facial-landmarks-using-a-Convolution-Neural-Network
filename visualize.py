from torchvision.transforms import Compose, Normalize

import transforms
from transforms import *
from utility import *
from dataset import FaceAlignmentDataset
from torch.utils.data import DataLoader as Dataloader

import matplotlib.pyplot as plt

face_dataset = FaceAlignmentDataset('training_images_full.npz', transform=Compose([ToTensor()]))
dataloader = Dataloader(face_dataset, batch_size=1, shuffle=True, num_workers=0)
mean, std = get_mean_and_std_of_dataloader(dataloader)
print(mean, std)
face_dataset = FaceAlignmentDataset('training_images_full.npz', transform=Compose([Downscale(2), ToTensor(), Normalise(mean, std)]))
dataloader = Dataloader(face_dataset, batch_size=4, shuffle=True, num_workers=0)
mean, std = get_mean_and_std_of_dataloader(dataloader)
print(mean, std)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['points'].size())
    image_batch = sample_batched['image']
    points_batch = sample_batched['points']

    if i_batch == 0:
        visualise_predicted_pts_from_tensor_batch(image_batch, points_batch, points_batch)
        print(image_batch[0])
        break
