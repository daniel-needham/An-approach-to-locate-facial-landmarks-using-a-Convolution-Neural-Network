from torchvision.transforms import Compose, RandomVerticalFlip

import transforms
from transforms import *
from utility import *
from model import ConvNet
from dataset import FaceAlignmentDataset
from torch.utils.data import DataLoader as Dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/face_alignment_experiment_1')

# hyperparameters
epochs = 1
learning_rate = 0.01
momentum = 0.9
batch_size = 4
scale_down = 2
image_dims = int(256 / scale_down)
num_of_points = 44

train_mean = torch.tensor([119.4163,  95.7439,  83.4596])
train_std = torch.tensor([77.5477, 68.5109, 66.8811])
test_mean = torch.tensor([122.3381,  97.8966,  84.6461])
test_std = torch.tensor([77.6566, 68.7760, 66.5846])

train_dataset = FaceAlignmentDataset('training_images_full.npz', train=True, transform=Compose([Downscale(2),ToTensor(), Normalise(train_mean, train_std)]))
test_dataset = FaceAlignmentDataset('training_images_full.npz', train=False, transform=Compose([Downscale(2), ToTensor(), Normalise(test_mean, test_std)]))

train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# define our model
model = ConvNet(image_dims, num_of_points)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
model.train() # Set the model to training mode
running_loss = 0.0
for epoch in range(epochs):
    for batch_i, batch in enumerate(train_loader):
        inputs, targets = batch['image'], batch['points']
        targets = flatten_coords_in_batch_tensor(targets)
        optimizer.zero_grad() # Zero the gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Compute the loss
        loss.backward() # Backward pass
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # Clip the gradients
        optimizer.step() # Update the model weights

        running_loss += loss.item()
        if batch_i % 100 == 9:  # every 10 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(train_loader) + batch_i)

            running_loss = 0.0
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Testing loop
model.eval() # Set the model to evaluation mode
total_loss = 0.0
with torch.no_grad():
    for batch_i, batch in enumerate(test_loader):
        inputs, targets, original_inputs = batch['image'], batch['points'], batch['original_image']
        targets = flatten_coords_in_batch_tensor(targets)
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Compute the loss
        total_loss += loss.item()
        # Visualize the 50th batch
        if batch_i % 50 == 0:
            targets = unflatten_coords_in_batch_tensor(targets, num_of_points)
            outputs = unflatten_coords_in_batch_tensor(outputs, num_of_points)
            visualise_predicted_pts_from_tensor_batch(original_inputs, targets, outputs)


    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

#todo
# - normalize the data (mean and std), possibly subtracting the mean face from each image
# - rotations
# -

