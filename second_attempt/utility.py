import matplotlib.pyplot as plt
import numpy as np

def display(image, landmarks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='X', c='r')
    plt.show()

def display_from_normalised(image,landmarks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    landmarks = unnormalise(image, landmarks)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='X', c='r')
    plt.title('Predicted Landmarks on an example image')
    plt.xlabel('X Dimension')
    plt.ylabel('Y Dimension')
    plt.show()

def display_both_points_from_normal(image, predict, landmarks, norm=True, image_tensor=True):
    plt.figure(figsize=(10, 10))
    if image_tensor:
        image = image.numpy().squeeze()

    plt.imshow(image, cmap='gray')
    if norm:
        predict = unnormalise(image, predict)
        landmarks = unnormalise(image, landmarks)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=20, marker='x', c='b', label='Actual')
    plt.scatter(predict[:, 0], predict[:, 1], s=20, marker='x', c='r', label='Predicted')
    plt.title(f'Predicted vs Actual Landmarks with a mean euclidean distance of: {euclid_dist(predict,landmarks):.4f} pixels')
    plt.xlabel('X Dimension')
    plt.ylabel('Y Dimension')
    plt.legend(loc='best')
    plt.show()


def unnormalise(image, landmarks):
    """
    :param image: image tensor
    :param landmarks: image landmarks
    :return: unnormalised landmarks scaled to the image size
    """
    landmarks = landmarks.numpy()
    landmarks = (landmarks + 0.5) * image.shape[1]
    return landmarks

def plot_train_and_valid(epochs, train_loss_epoch, valid_loss_epoch):
    """
    Plot the training and validation loss
    :param epochs: The number of epochs
    :param train_loss_epoch: Train loss per epoch
    :param valid_loss_epoch: Validation loss per epoch
    :return: None
    """
    plt.figure(figsize=(10, 10))
    epoch = range(1, epochs + 1)
    plt.plot(epoch, train_loss_epoch, label='Training Loss')
    plt.plot(epoch, valid_loss_epoch, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(np.arange(0, epochs + 1, 1))

    # Display the plot
    plt.legend(loc='best')
    plt.show()

def plot_mean_euclid_dists(mean_euclid_distance):
    """
    Plot the mean euclidean distance
    :param mean_euclid_distance: The mean euclidean distance
    :return: None
    """
    plt.figure(figsize=(10, 10))
    plt.hist(mean_euclid_distance, bins=20)

    plt.title('Mean Euclidean Distance between Predicted and Actual Landmarks for images in the test set')
    plt.xlabel('Mean Euclidean Distance')
    plt.ylabel('Frequency')

    plt.show()

def euclid_dist(pred_pts, gt_pts):
    """
    Calculate the euclidean distance between pairs of points
    :param pred_pts: The predicted points
    :param gt_pts: The ground truth points
    :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
    """
    import numpy as np
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.mean(np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1)))


# provided functions
def extract_subset_of_points(pts):
  indices = (20, 29, 16, 32, 38)
  if len(pts.shape) == 3:
    return pts[:, indices, :]
  elif len(pts.shape) == 2:
    return pts[indices, :]

def visualise_pts(img, pts):
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r', ms=7)
  plt.show()


def save_as_csv(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==44*2, 'wrong number of points provided. There should be 34 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
