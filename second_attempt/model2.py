import torch.nn as nn
import torchvision.models as models

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,88)

    def forward(self,x):
        x = self.model(x)
        return x