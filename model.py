import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, image_dim, num_of_points):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(20480, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_of_points * 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.relu(out)


        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out