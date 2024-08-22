import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self,num_classes):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,stride=4,padding=0)
        self.maxPool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(96,256,5,stride=1,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,stride=1,padding=1)
        self.fc1 = nn.Linear(256*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(384)
        self.d = nn.Dropout(0.5)

    def forward(self,x):
        x = self.maxPool(F.relu(self.b1(self.conv1(x))))
        x = self.maxPool(F.relu(self.b2(self.conv2(x))))
        x = F.relu(self.b3(self.conv3(x)))
        x = F.relu(self.b3(self.conv4(x)))
        x = self.maxPool(F.relu(self.b2(self.conv5(x))))
        # x = torch.flatten(x,1)
        x = x.reshape(x.size(0),-1)
        x = F.relu(self.fc1(self.d(x)))
        x = F.relu(self.fc2(self.d(x)))
        x = F.relu(self.fc3(x))
        return x
