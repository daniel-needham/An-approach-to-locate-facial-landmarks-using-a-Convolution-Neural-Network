import cv2
import torch
import numpy as np
from torchvision.transforms import Normalize

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

class Normalise(object):
    """Normalise RGB image"""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
       #normalise rgb image
        image = sample['image']
        points = sample['points']
        image = Normalize(self.mean, self.std)(image)
        return {'image': image,
                'points': points}


class Greyscale(object):
    """Convert image to greyscale."""
    def __call__(self, sample):
        image = sample['image']
        points = sample['points']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(image.shape)
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
            #numpy flip
            image = np.flip(image,axis=0)
            image = np.ascontiguousarray(image)
            #flip the points vertically
            points[:,1] = image.shape[0] - points[:,1]
        return {'image': image,
                'points': points}

class RandomRotate(object):
    def __init__(self, max_angle=0):
        self.max_angle = max_angle

    def rotate_points(self, center, points, angle):
        x_center, y_center = center
        angle_rad = -angle * np.pi / 180
        points = points.reshape(-1, 2)
        points = np.copy(points)
        points[:, 0] = x_center + np.cos(angle_rad) * (points[:, 0] - x_center) - np.sin(angle_rad) * (points[:, 1] - y_center)
        points[:, 1] = y_center + np.sin(angle_rad) * (points[:, 0] - x_center) + np.cos(angle_rad) * (points[:, 1] - y_center)
        return points

    def __call__(self, sample):
        image = sample['image']
        points = sample['points']
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        center = (image.shape[0]//2, image.shape[1]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        points = self.rotate_points(center, points, angle)
        return {'image': image,
                'points': points}
