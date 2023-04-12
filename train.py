from torchvision.transforms import Compose, RandomVerticalFlip
from transforms import *
from utility import *
from model import ConvNet
from dataset import FaceAlignmentDataset
from torch.utils.data import DataLoader as Dataloader
import torch
import torch.nn as nn
import torch.optim as optim

# hyperparameters
epochs = 10
learning_rate = 0.01
momentum = 0.9
batch_size = 4
scale_down = 2
image_dims = int(256 / scale_down)
num_of_points = 44

train_dataset = FaceAlignmentDataset('training_images_full.npz', train=True, transform=Compose([Downscale(scale_down),ToTensor(), RandomVerticalFlip()]))
test_dataset = FaceAlignmentDataset('training_images_full.npz', train=False, transform=Compose([Downscale(scale_down),ToTensor()]))

train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Dataloader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# define our model
model = ConvNet(image_dims, num_of_points)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
model.train() # Set the model to training mode

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

        if batch_i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Testing loop
model.eval() # Set the model to evaluation mode
total_loss = 0.0
with torch.no_grad():
    for batch_i, batch in enumerate(test_loader):
        inputs, targets = batch['image'], batch['points']
        targets = flatten_coords_in_batch_tensor(targets)
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets) # Compute the loss
        total_loss += loss.item()
        if batch_i % 50 == 0:
            targets = unflatten_coords_in_batch_tensor(targets, num_of_points)#
            outputs = unflatten_coords_in_batch_tensor(outputs, num_of_points)
            visualise_predicted_pts_from_tensor_batch(inputs, targets, outputs)


    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

