import torch
import torch.nn as nn 

class BlockA(nn.Module):
    def __init__(self, in_ch, out_ch, identity_downsample=None, stride=1, cardinality=32):
        super(BlockA, self).__init__()
        self.identity_downsample = identity_downsample
        self.cardinality = cardinality
        self.expansion = 2 # 'each time when the spatial map is downsampled by a factor of 2 the width of the block is multiplied by a factor of 2'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch*self.expansion, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch*self.expansion),
        )
        self.relu = nn.ReLU()

    def forward(self, x_input):
        # y = torch.zeros((x.shape))
        y = None

        for i in range(self.cardinality):
            x = x_input.clone()
            identity = x.clone()
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)

            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)

            x += identity
            x = self.relu(x)
            if i==0:
                y = torch.zeros((x.shape))
            y += x
        return y
    

class BlockB(nn.Module):
    def __init__(self, in_ch, out_ch, identity_downsample=None, stride=1, cardinality=32):
        super(BlockB, self).__init__()
        self.identity_downsample = identity_downsample
        self.cardinality = cardinality
        self.expansion = 2 # 'each time when the spatial map is downsampled by a factor of 2 the width of the block is multiplied by a factor of 2'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv_end = nn.Conv2d(in_channels=out_ch*self.cardinality, out_channels=out_ch*self.expansion, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_features=out_ch*self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x_input):
        y = []
        identity = x_input.clone()
        for i in range(self.cardinality):
            x = x_input.clone()
            x = self.conv1(x)
            x = self.conv2(x)
            # print(x.shape)
            y.append(x)
        
        z = torch.cat(y,dim=1)
        z = self.conv_end(z)
        z = self.bn(z)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        z = self.relu(z)
        z += identity
        return z
    
class BlockC(nn.Module):
    def __init__(self,in_ch, out_ch, identity_downsample=None, stride=1, cardinality=32):
        super(BlockC, self).__init__()
        self.identity_downsample = identity_downsample
        self.cardinality = cardinality
        self.expansion = 2 # 'each time when the spatial map is downsampled by a factor of 2 the width of the block is multiplied by a factor of 2'
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,stride=stride, padding=1, groups=self.cardinality),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch*self.expansion, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch*self.expansion),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x



class ResNeXt(nn.Module):
    def __init__(self, block, layer, image_channels, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.in_channels = 128
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=128,kernel_size=7 ,stride=2, padding=3)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1 )

        self.layer1 = self._make_layer(block,layer[0],out_channels=128,stride = 1)
        self.layer2 = self._make_layer(block,layer[1],out_channels=256,stride = 2)
        self.layer3 = self._make_layer(block,layer[2],out_channels=512,stride = 2)
        self.layer4 = self._make_layer(block,layer[3],out_channels=1024,stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
         
        if self.in_channels != 2*out_channels:
            (self.in_channels, out_channels)
            identity_downsample = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels*2, kernel_size=1, stride=stride), 
                                                nn.BatchNorm2d(out_channels*2))

        layers.append(block(self.in_channels, out_channels, identity_downsample,stride=stride))
        self.in_channels = out_channels*2

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
def ResNeXt50A(image_channels=3, num_classes=1000):
    return ResNeXt(BlockA, [3,4,6,3], image_channels, num_classes)

def ResNeXt50B(image_channels=3, num_classes=1000):
    return ResNeXt(BlockB, [3,4,6,3], image_channels, num_classes)

def ResNeXt50C(image_channels=3, num_classes=1000):
    return ResNeXt(BlockC, [3,4,6,3], image_channels, num_classes)