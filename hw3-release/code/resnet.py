import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(num_channels)
        conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(num_channels)       

        self.fx = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        final_active = nn.ReLU()
        return final_active(self.fx(x) + x)


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.block = Block(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.block(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x