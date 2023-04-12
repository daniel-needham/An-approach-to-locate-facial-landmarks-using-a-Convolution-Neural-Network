from torchvision.transforms import Compose
from transforms import *
from utility import *
from dataset import FaceAlignmentDataset
from torch.utils.data import DataLoader as Dataloader

import matplotlib.pyplot as plt

face_dataset = FaceAlignmentDataset('training_images_full.npz', transform=Compose([Downscale(4),ToTensor()]))

# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#     print(i, sample['image'].size(), sample['points'].size())
#
#     visualise_pts_from_tensor(sample['image'], sample['points'])
#
#
#     if i == 3:
#         plt.show()
#         break


dataloader = Dataloader(face_dataset, batch_size=4, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['points'].size())
    image_batch = sample_batched['image']
    points_batch = sample_batched['points']

    points_batch = flatten_coords_in_batch_tensor(points_batch)
    print(i_batch, image_batch.size(), points_batch.size())
    points_batch = unflatten_coords_in_batch_tensor(points_batch, 44)
    print(i_batch, image_batch.size(), points_batch.size())
    if i_batch == 0:
        visualise_predicted_pts_from_tensor_batch(image_batch,points_batch, points_batch)
        break
