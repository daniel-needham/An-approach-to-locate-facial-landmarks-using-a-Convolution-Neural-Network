import torch
from torch.utils.data import Dataset
import numpy as np


class FaceAlignmentDataset(Dataset):
    def __init__(self, data_filename, train=False, transform=None):
        data = np.load(data_filename, allow_pickle=True)
        if train:
            images = data['images'][:int(len(data['images']) * 0.8)]
            points = data['points'][:int(len(data['points']) * 0.8)]
        else:
            images = data['images'][int(len(data['images']) * 0.8):]
            points = data['points'][int(len(data['points']) * 0.8):]

        self.data = {'images': images, 'points': points}
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Extract the images
        image = self.data['images'][index]
        # Extract data points
        points = self.data['points'][index]

        sample = {'image': image, 'points': points}

        # Apply transformations
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data['images'])