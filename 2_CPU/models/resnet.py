import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        # If output size is not the same as input size, adjust with 1x1 conv
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        conv1 = self.residual_function[0](x)
        bn1 = self.residual_function[1](conv1)
        relu = self.residual_function[2](bn1)
        conv2 = self.residual_function[3](relu)
        bn2 = self.residual_function[4](conv2)
        shortcut = self.shortcut(x)
        out = F.relu(bn2 + shortcut)
        return out, [relu, bn2]

DIM = 64

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super().__init__()
        self.in_channels = DIM 

        self.conv1 = nn.Conv2d(num_channels, DIM, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(DIM)

        # Dynamically add blocks to the layer
        self.layer1 = self._make_layer(block, DIM, num_blocks[0])
        self.layer2 = self._make_layer(block, DIM*2, num_blocks[1])
        self.layer3 = self._make_layer(block, DIM*4, num_blocks[2])
        self.layer4 = self._make_layer(block, DIM*8, num_blocks[3])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(DIM*8 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks):
        layers = []
        strides = [2] + (num_blocks-1)*[1]
        for i in range(num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=strides[i]))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        intermediate_results = []

        out = self.conv1(x)
        out = self.bn1(out)
        intermediate_results.append(out.detach().numpy())
        out = F.relu(out)

        for i, layer in enumerate(self.layer1):
            out, [conv1, conv2] = layer(out)
            intermediate_results.append(conv1.detach().numpy())
            if i == 0:
                intermediate_results.append(conv2.detach().numpy())
            intermediate_results.append(out.detach().numpy())
            
        for i, layer in enumerate(self.layer2):
            out, [conv1, conv2]= layer(out)
            intermediate_results.append(conv1.detach().numpy())
            if i == 0:
                intermediate_results.append(conv2.detach().numpy())
            intermediate_results.append(out.detach().numpy())
        
        for i, layer in enumerate(self.layer3):
            out, [conv1, conv2]= layer(out)
            intermediate_results.append(conv1.detach().numpy())
            if i == 0:
                intermediate_results.append(conv2.detach().numpy())
            intermediate_results.append(out.detach().numpy())
        
        for i, layer in enumerate(self.layer4):
            out, [conv1, conv2]= layer(out)
            intermediate_results.append(conv1.detach().numpy())
            if i == 0:
                intermediate_results.append(conv2.detach().numpy())
            intermediate_results.append(out.detach().numpy())

        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        intermediate_results.append(out.detach().numpy())

        assert len(intermediate_results) == 22, f"Expected 22 intermediate results, got {len(intermediate_results)}"

        return out, intermediate_results 

def resnet14(num_classes, num_channels):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, num_channels)

def resnet18(num_classes, num_channels):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)

def resnet34(num_classes, num_channels):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_channels)

# if __name__ == "__main__":
#     x = torch.randn(1,3,32,32)
#     model = resnet18(10, 3)
#     model.eval()
#     y, intermediate = model(x)
#     print("intermediate results: ", len(intermediate)) 
#     print(intermediate)