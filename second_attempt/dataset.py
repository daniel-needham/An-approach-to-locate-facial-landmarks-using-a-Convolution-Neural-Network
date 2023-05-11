import torch
import numpy as np
from torch.utils.data import Dataset

class FaceAlignmentDataset(Dataset):
    def __init__(self, data_filename, transform=None, length=None):
        data = np.load(data_filename, allow_pickle=True)

        self.images = data['images']
        if data.__contains__('points'):
            self.landmarks = data['points']
        else:
            self.landmarks = np.zeros((len(self.images)))
        self.transform = transform

        if length is not None:
            self.images = self.images[:length]
            self.landmarks = self.landmarks[:length]

        assert len(self.images) == len(self.landmarks)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.images[index]
        image_original = image.copy()
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks)

        landmarks = landmarks - 0.5

        return image, image_original, landmarks


    def __len__(self):
        return len(self.images)