import torch
import numpy as np

class Downscale(object):
    """Downscale the image by a factor of 2."""
    def __init__(self, factor=2):
        assert isinstance(factor, int), 'factor must be an integer'
        assert factor > 0, 'factor must be greater than 0'
        assert factor % 2 == 0, 'factor must be a multiple of 2'
        self.factor = factor
    def __call__(self, sample):
        image = sample['image']
        points = sample['points']
        image = image[::self.factor, ::self.factor, :]
        points = points / self.factor
        return {'image': image,
                'points': points}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        points = sample['points']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        image_torch = torch.from_numpy(image)
        points_torch = torch.from_numpy(points)
        image_torch = image_torch.float()
        points_torch = points_torch.float()
        return {'image': image_torch,
                'points': points_torch}
class RandomVerticalFlip(object):
    """Randomly flip the image vertically."""
    def __call__(self, sample):
        image = sample['image']
        points = sample['points']
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            points[1::2] = image.shape[0] - points[1::2]
        return {'image': image,
                'points': points}