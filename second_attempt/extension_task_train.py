import torch.utils.data
from dataset import FaceAlignmentDataset
from transforms import Transforms
import matplotlib.pyplot as plt
from utility import *
from torch.utils.data import DataLoader as Dataloader
from model import ConvNet
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


# hyperparameters
epochs = 15
learning_rate = 0.1
momentum = 0.9
batch_size = 1
scale_down = 2
image_dims = int(256 / scale_down)
num_of_points = 44

#setup datasets
original_dataset = FaceAlignmentDataset('training_images_full.npz',transform=Transforms())
extended_dataset = FaceAlignmentDataset('training_images_subset_predictions.npz',transform=Transforms())

train_dataset, test_dataset = torch.utils.data.random_split(original_dataset, [0.8,0.2])

complete_dataset = torch.utils.data.ConcatDataset([train_dataset, extended_dataset])

train_dataset, val_dataset = torch.utils.data.random_split(complete_dataset, [0.9, 0.1])

#setup dataloaders
train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = Dataloader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)



model = ConvNet(image_dims, num_of_points)
#model = ConvNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) #, weight_decay=1e-5
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# set min loss
loss_min = np.inf

# set up arrays for running train/validation loss
train_loss_epoch = []
valid_loss_epoch = []

for epoch in range(1, epochs + 1):
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    model.train()
    for batch_i, batch in enumerate(train_loader):
        image, image_original, landmarks = batch
        landmarks = landmarks.view(landmarks.size(0), -1)
        outputs = model(image)

        #clear gradients
        optimizer.zero_grad()

        #find loss
        loss = criterion(outputs, landmarks)

        #back prop pass
        loss.backward()

        #clip gradients
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # Clip the gradients

        #update params
        optimizer.step()

        loss_train += loss.item()
        running_loss = loss_train / (batch_i + 1)
        if batch_i % 100 == 9:  # every 100 mini-batches...
            print(f'Epoch: [{epoch}/{epochs}], Batch: [{batch_i + 1}/{len(train_loader) + 1}] \t\t Training Loss: {running_loss:.9f}')

    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(val_loader):
            image, image_original, landmarks = batch
            landmarks = landmarks.view(landmarks.size(0), -1)
            outputs = model(image)

            # find loss
            loss_valid_step = criterion(outputs, landmarks)


            loss_valid += loss_valid_step.item()
            running_loss = loss_valid / (batch_i + 1)

            if batch_i % 10 == 9:  # every 100 mini-batches...
                print(f'Batch: [{batch_i + 1}/{len(val_loader) + 1}] \t\t Validation Loss: {running_loss:.9f}')

    loss_train /= len(train_loader)
    loss_valid /= len(val_loader)

    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.9f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')

    # add to loss arrays
    train_loss_epoch.append(loss_train)
    valid_loss_epoch.append(loss_valid)

    # save model if validation loss has decreased
    if loss_valid < loss_min:
        loss_min = loss_valid
        print(os.getcwd())
        torch.save(model.state_dict(), os.getcwd() + '/face_landmarks_extended_dataset.pth')
        print("\nMinimum Validation Loss of {:.9f} at epoch {}/{}".format(loss_min, epoch, epochs))
        print('Model Saved\n')

print('Finished Training')
print('Starting Testing')

#
mean_euclidean_distance = []

model.eval()
total_loss = 0
with torch.no_grad():
    for batch_i, batch in enumerate(test_loader):
        image, image_original, landmarks = batch
        landmarks = landmarks.view(landmarks.size(0), -1)
        outputs = model(image)

        # find loss
        loss_test_step = criterion(outputs, landmarks)

        total_loss += loss_test_step.item()
        running_loss = total_loss / (batch_i + 1)

        outputs = outputs.view(outputs.size(0), -1, 2)
        landmarks = landmarks.view(landmarks.size(0), -1, 2)

        # every 100 mini-batches... display image with predicted and actual landmarks
        # if batch_i % 100 == 0:
        #     for i in range(outputs.shape[0]):
        #         display_both_points_from_normal(image[i], outputs[i], landmarks[i])


        # calculate euclidean distance
        for i in range(outputs.shape[0]):
            unnorm_outputs = unnormalise(image[i], outputs[i])
            unnorm_landmarks = unnormalise(image[i], landmarks[i])
            ed = euclid_dist(unnorm_outputs, unnorm_landmarks)

            if ed > 10:
                display_both_points_from_normal(image[i], outputs[i], landmarks[i])
            if ed < 1.5:
                display_both_points_from_normal(image[i], outputs[i], landmarks[i])
            mean_euclidean_distance.append(ed)

        print(f'Batch: {batch_i + 1} \t\t Test Loss: {running_loss:.9f}')

    total_loss /= len(test_loader)
    print('\n--------------------------------------------------')
    print('Test Loss: {:.4f}'.format(total_loss))

# plot mean euclidean distance for images in test set
plot_mean_euclid_dists(mean_euclidean_distance)
print('Mean Euclidean Distance: {:.4f}'.format(np.mean(mean_euclidean_distance)))

# plot train and validation loss
plot_train_and_valid(epochs, train_loss_epoch, valid_loss_epoch)