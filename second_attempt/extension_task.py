import numpy as np
from utility import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

# load the full and subset training images
full_data = np.load('training_images_full.npz', allow_pickle=True)
subset_data = np.load('training_images_subset.npz', allow_pickle=True)

full_images = full_data['images']
full_landmarks = full_data['points']

subset_images = subset_data['images']
subset_landmarks = subset_data['points']

# extract the subset of points from the full dataset

full_data_subset_landmarks = extract_subset_of_points(full_landmarks)

full_data_subset_landmarks = full_data_subset_landmarks.reshape(full_data_subset_landmarks.shape[0], -1)
full_landmarks = full_landmarks.reshape(full_landmarks.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(full_data_subset_landmarks, full_landmarks, test_size=0.2, shuffle=False)

model = LinearRegression(normalize=True, fit_intercept=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, predictions))

# generate 44 facial landmarks for the training_images_subset.npz dataset
subset_landmarks = subset_landmarks.reshape(subset_landmarks.shape[0], -1)
y = model.predict(subset_landmarks)

# reshape the predictions to be in the same format as the original dataset
y = y.reshape(subset_landmarks.shape[0], 44, 2)
subset_landmarks = subset_landmarks.reshape(subset_landmarks.shape[0], 5, 2)

# save the predictions to a file
np.savez('training_images_subset_predictions.npz', images=subset_images, points=y)

# plot the predictions of the 44 points vs the ground truth
ex = full_images[0]
ex_gt = full_landmarks[0]
ex_gt = ex_gt.reshape(44, 2)

ex_y = model.predict(full_data_subset_landmarks[0].reshape(1,-1))
ex_y = ex_y.reshape(1, 44, 2)

visualise_pts(ex, ex_gt)
visualise_pts(ex, ex_y[0])
display_both_points_from_normal(ex, ex_y[0], ex_gt, norm=False, image_tensor=False)

# combine training_images_subset_predictions.npz and training_images_full.npz into a single dataset
print(full_images.shape)
print(subset_images.shape)

combined_images = np.concatenate((full_images, subset_images), axis=0)
print(f'combined image shape: {combined_images.shape}')

print(full_landmarks.shape)
print(y.shape)

full_landmarks = full_landmarks.reshape(full_landmarks.shape[0], 44, 2)
combined_landmarks = np.concatenate((full_landmarks, y), axis=0)

print(f'combined landmarks shape: {combined_landmarks.shape}')

# save the combined dataset
np.savez('training_images_combined.npz', images=combined_images, points=combined_landmarks)