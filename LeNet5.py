import torch
import torch.nn as nn
import torch.nn.functional as F
'''
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
'''

### LaNet for MNIST dataset(digit classification)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__() #(the class from which you wanted the "super" to, and the instance where you'd call the method)
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        # self.avgpool2 = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
