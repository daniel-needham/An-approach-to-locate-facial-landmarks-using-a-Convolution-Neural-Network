import matplotlib.pyplot as plt
import numpy as np

def display(image, landmarks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8, marker='X', c='r')
    plt.show()

def display_from_normalised(image,landmarks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    landmarks = unnormalise(image, landmarks)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=8, marker='X', c='r')
    plt.show()

def unnormalise(image, landmarks):
    landmarks = landmarks.numpy()
    landmarks = (landmarks + 0.5) * image.shape[1]
    return landmarks