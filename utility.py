import numpy as np
import torch
import matplotlib.pyplot as plt


def visualise_pts(img, pts):
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r', ms=7)
  plt.show()

def visualise_pts_from_tensor(img, pts):
  import matplotlib.pyplot as plt
  print(img.shape)
  img = img.numpy().transpose((0,2, 3, 1))
  img = np.squeeze(img)
  #convert img to int
  img = img.astype(np.uint8)
  plt.imshow(img)
  pts = pts.numpy()
  pts = np.squeeze(pts)
  plt.plot(pts[:, 0], pts[:, 1], '+r', ms=7)
  plt.show()

def visualise_predicted_pts_from_tensor_batch(img, pts, y_pts):
    import matplotlib.pyplot as plt
    #transpose batch of img to (batch, channel, height, width)
    img = img.numpy().transpose((0, 2, 3, 1))
    #convert img to int
    img = img.astype(np.uint8)
    pts = pts.numpy()
    y_pts = y_pts.numpy()
    fig = plt.figure(figsize=(8, 8))
    #for each image in the batch
    for i in range(img.shape[0]):
        #show image and points as subplot
        fig.add_subplot(2, img.shape[0], i+1)
        plt.title('truth')
        plt.imshow(img[i])
        plt.plot(pts[i, :, 0], pts[i, :, 1], '+r', ms=7)
        # plot on the second row
        fig.add_subplot(2, img.shape[0], i + 5)
        plt.title('pred')
        plt.imshow(img[i])
        plt.plot(y_pts[i, :, 0], y_pts[i, :, 1], '+y', ms=7)
    plt.show()

def flatten_coords_in_batch_tensor(batch):
    return batch.view(batch.shape[0], -1)

def unflatten_coords_in_batch_tensor(batch, num_of_points):
    # unflatten the batch of points
    return batch.view(batch.shape[0], num_of_points, -1)


def get_mean_and_std_of_dataloader(dataloader):
    channel_sum, channel_squared_sum, num_batches = 0, 0, 0
    for data in dataloader:
        image = data['image']
        channel_sum += torch.mean(image, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(image ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channel_sum / num_batches
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
def save_as_csv(points, location='.'):
  """
  Save the points out as a .csv file
  :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
  :param location: Directory to save results.csv in. Default to current working directory
  """
  assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
  assert np.prod(points.shape[
                 1:]) == 44 * 2, 'wrong number of points provided. There should be 34 points with 2 values (x,y) per point'
  np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

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
  return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))

def extract_subset_of_points(pts):
  indices = (20, 29, 16, 32, 38)
  if len(pts.shape) == 3:
    return pts[:, indices, :]
  elif len(pts.shape) == 2:
    return pts[indices, :]
