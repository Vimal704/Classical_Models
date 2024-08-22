import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg16(nn.Module):
    def __init__(self,num_classes=4):
        super(Vgg16,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12*12*512,4096),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        print(x.shape)
        x = torch.flatten(x,1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x