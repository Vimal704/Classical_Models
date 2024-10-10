import torch
import torch.nn as nn 
import torchvision.transforms.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class UNet(nn.Module):
    def __init__(self, channels = [64,128,256,512]):
        super(UNet, self).__init__()
        # self.layers = layers
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.Tconv = nn.ConvTranspose2d(input_channels)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        input_channels = 3
        for i in channels:
            self.downs.append(Block(input_channels, i))
            input_channels = i
        
        self.trough = Block(input_channels, input_channels*2)

        input_channels = input_channels*2
        for i in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(in_channels=input_channels, out_channels=i,kernel_size=2, stride=2))
            self.ups.append(Block(input_channels, i))
            input_channels=i

        self.final = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
        
    def forward(self,x):
        residuals = []
        for layer in self.downs:
            x = layer(x)
            residuals.append(x)
            x = self.pool(x)

        x = self.trough(x)
        residuals = residuals[::-1]
        
        for layer_no in range(0,len(self.ups),2):
            x = self.ups[layer_no](x)
            residual = residuals[layer_no//2]
            residual = F.resize(residual, size=x.shape[2:])
            x = torch.cat((residual, x), dim=1)
            x = self.ups[layer_no+1](x)
        
        x = self.final(x)
        return x

if __name__ == '__main__':
    x = torch.rand((5,3,572, 572))
    model = UNet()
    print(model(x).shape)




            
        