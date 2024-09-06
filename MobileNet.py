import torch 
import torch.nn as nn
import torch.nn.functional as F

class DepthSepConv(nn.Module):
    def __init__(self, in_cha, out_cha, stride=1, pad=1 ,alpha=1):
        super(DepthSepConv, self).__init__()
        self.depth = nn.Conv2d(in_channels=in_cha, out_channels=in_cha, kernel_size=3,stride=stride, padding=pad, groups=in_cha)
        self.batch_norm1 = nn.BatchNorm2d(in_cha)
        self.point = nn.Conv2d(in_channels=in_cha, out_channels=int(out_cha*alpha), kernel_size=1)
        self.batch_norm2 = nn.BatchNorm2d(int(out_cha*alpha))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.depth(x)))
        x = self.relu(self.batch_norm2(self.point(x)))
        return x

class MobileNet(nn.Module):
    def __init__(self, alpha=1, rho=1, num_output=1000):
        super(MobileNet, self).__init__()
        self.rho = rho
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.convb1 = DepthSepConv(32, 64, 1, 1, alpha)
        self.convb2 = DepthSepConv(64, 128, 2, 1, alpha)
        self.convb3 = DepthSepConv(128, 128, 1, 1, alpha)
        self.convb4 = DepthSepConv(128, 256, 2, 1, alpha)
        self.convb5 = DepthSepConv(256, 256, 1, 1, alpha)
        self.convb6 = DepthSepConv(256, 512, 2, 1, alpha)
        self.convb7 = DepthSepConv(512, 512, 1, 1, alpha)
        self.convb8 = DepthSepConv(512, 1024, 2, 1, alpha)
        self.convb9 = DepthSepConv(1024, 1024, 2, 1, alpha)
        self.avgpool = nn.AdaptiveMaxPool2d(output_size=1)
        self.fc = nn.Linear(in_features=1024, out_features=num_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        _, _, H, W = x.size()
        h_new = int(H*self.rho) 
        w_new = int(W*self.rho) 
        x = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)
        x = self.relu(self.bn(self.conv0(x)))
        x = self.convb1(x)
        x = self.convb2(x)
        x = self.convb3(x)
        x = self.convb4(x)
        x = self.convb5(x)
        x = self.convb6(x)
        x = self.convb7(self.convb7(self.convb7(self.convb7(self.convb7(x)))))
        x = self.convb8(x)
        x = self.convb9(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x
    
if __name__ == '__main__':
    x = torch.rand((3,3,224,224))
    model = MobileNet()
    print(torch.sum(model(x), dim=1))