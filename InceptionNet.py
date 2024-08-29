import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels,dim):
        super(InceptionModule, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, dim[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, dim[1], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(dim[1], dim[2], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, dim[3], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(dim[3], dim[4], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, dim[5], kernel_size=1, padding=0),
            nn.ReLU()
        )
        
    def forward(self,x):
        return torch.cat((self.layer1(x), self.layer2(x), self.layer3(x), self.layer4(x)),dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_channels=1000):
        super(GoogLeNet,self).__init__()
        dims = {'3a':[64, 96, 128, 16, 32, 32],
                '3b':[128, 128, 192, 32, 96, 64],
                '4a':[192, 96, 208, 16, 48, 64],
                '4b':[160, 112, 224, 24, 64, 64],
                '4c':[128, 128, 256, 24, 64, 64],
                '4d':[112, 144, 288, 32, 64, 64],
                '4e':[256, 160, 320, 32, 128, 128],
                '5a':[256, 160, 320, 32, 128, 128],
                '5b':[384, 192, 384, 48, 128, 128]
            }
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0,ceil_mode=True),
            nn.LocalResponseNorm(size=5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 192,kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,ceil_mode=True)
        self.inception3a = InceptionModule(192, dims['3a'])
        self.inception3b = InceptionModule(sum(dims['3a'])-dims['3a'][1]-dims['3a'][3], dims['3b'])
        self.inception4a = InceptionModule(sum(dims['3b'])-dims['3b'][1]-dims['3b'][3], dims['4a'])
        self.inception4b = InceptionModule(sum(dims['4a'])-dims['4a'][1]-dims['4a'][3], dims['4b'])
        self.inception4c = InceptionModule(sum(dims['4b'])-dims['4b'][1]-dims['4b'][3], dims['4c'])
        self.inception4d = InceptionModule(sum(dims['4c'])-dims['4c'][1]-dims['4c'][3], dims['4d'])
        self.inception4e = InceptionModule(sum(dims['4d'])-dims['4d'][1]-dims['4d'][3], dims['4e'])
        self.inception5a = InceptionModule(sum(dims['4e'])-dims['4e'][1]-dims['4e'][3], dims['5a'])
        self.inception5b = InceptionModule(sum(dims['5a'])-dims['5a'][1]-dims['5a'][3], dims['5b'])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.aux0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_channels, bias=True),
            nn.Softmax(dim=1)
        )
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(528, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_channels, bias=True),
            nn.Softmax(dim=1)
        )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_channels, bias=True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception4a(x)
        out0 = self.aux0(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        out1 = self.aux1(x)
        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        out2 = self.aux2(x)

        return out2,out1,out0
        # return out2
    

if __name__ == '__main__':
    x = torch.rand((3,3,224,224))
    model = GoogLeNet()
    print(model(x)[0].shape)
    print(model(x)[1].shape)
    print(model(x)[2].shape)