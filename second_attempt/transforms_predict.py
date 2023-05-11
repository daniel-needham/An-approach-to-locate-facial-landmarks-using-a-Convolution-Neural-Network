import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import PIL


class Transforms():
    def __init__(self):
        pass

    def greyscale(self, image):
        # Convert to greyscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def downscale(self, image, landmarks, scale):
        assert scale % 2 == 0, "Scale must be divisible by 2"
        image = image[::scale, ::scale]
        landmarks = landmarks / scale
        return image, landmarks

    def normalise_landmarks(self, image, landmarks):
        norm = landmarks / torch.tensor([image.shape[1], image.shape[1]])
        return norm

    def colour_jitter(self, image):
        image = PIL.Image.fromarray(image)
        colour_jitter = ColorJitter(brightness=0.3, contrast=0.3)
        image = colour_jitter(image)
        image = np.array(image)
        return image

    def rotate_points(self, center, points, angle):
        x_center, y_center = center
        angle_rad = -angle * np.pi / 180
        points = points.reshape(-1, 2)
        points = np.copy(points)
        points[:, 0] = x_center + np.cos(angle_rad) * (points[:, 0] - x_center) - np.sin(angle_rad) * (points[:, 1] - y_center)
        points[:, 1] = y_center + np.sin(angle_rad) * (points[:, 0] - x_center) + np.cos(angle_rad) * (points[:, 1] - y_center)
        return points

    def random_rotate(self, image, landmarks, max_angle=10):
        angle = np.random.uniform(-max_angle, max_angle)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        landmarks = self.rotate_points(center, landmarks, angle)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return image, landmarks

    def random_vertical_flip(self, image, landmarks):
        if np.random.rand() > 0.5:
            #numpy flip
            image = np.flip(image,axis=0)
            image = np.ascontiguousarray(image)
            #flip the points vertically
            landmarks[:,1] = image.shape[0] - landmarks[:,1]
        return image, landmarks

    def random_horizontal_flip(self, image, landmarks):
        if np.random.rand() > 0.1:
            #numpy flip
            image = np.flip(image,axis=1)
            image = np.ascontiguousarray(image)
            #flip the points horizontally
            landmarks[:,0] = image.shape[1] - landmarks[:,0]
        return image, landmarks

    def gaussian_blur(self, image):
        image = gaussian_filter(image, sigma=1.5)
        return image
    def __call__(self, image, landmarks):
        image = self.greyscale(image)
        image, landmarks = self.downscale(image, landmarks, 2)
        image = self.gaussian_blur(image)
        #image = self.colour_jitter(image)
        #image, landmarks = self.random_horizontal_flip(image, landmarks)
        #image, landmarks = self.random_rotate(image, landmarks)


        image = TF.to_tensor(image)
        landmarks = self.normalise_landmarks(image, landmarks)
        image = TF.normalize(image, [0.5], [0.5])
        image = image.float()
        landmarks = landmarks.float()

        return image, landmarks
