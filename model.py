import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, image_dim, num_of_points):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16384, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_of_points * 2)

    def forward(self, x):
        # layer 1
        out = self.relu(self.max_pool(self.conv1(x)))

        # layer 2
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out  = self.max_pool(out)

        # layer 3
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(out))
        out  = self.max_pool(out)

        # layer 4
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))
        out  = self.max_pool(out)

        # layer 5
        out = self.relu(self.conv8(out))

        # layer 6
        out = out.reshape(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out