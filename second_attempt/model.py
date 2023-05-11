
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, image_dim, num_of_points):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4096, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_of_points * 2)
        self.sig = nn.Sigmoid(, in
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        # layer 1
        out = self.relu(self.max_pool(self.conv1(x)))
        out = nn.BatchNorm2d(64)(out)

        # layer 2
        out = self.relu(self.conv2(out))
        out = nn.BatchNorm2d(192)(out)
        out  = self.max_pool(out)



        # layer 3
        out = self.relu(self.conv3(out))
        out = nn.BatchNorm2d(384)(out)

        # layer 4
        out = self.relu(self.conv4(out))
        out = nn.BatchNorm2d(256)(out)
        out  = self.max_pool(out)

        # layer 5
        out = self.relu(self.conv5(out))
        out = nn.BatchNorm2d(256)(out)


        # layer 6
        out = out.reshape(out.size(0), -1)
        out = self.sig(self.fc1(out))
        out = self.fc2(out)
        return out