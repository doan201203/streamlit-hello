# Define the Residual Block
import torch
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels are different, or stride > 1,
        # we need a shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# Define the modified network with Dropout and Residual Block
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu_conv1 = nn.ReLU()
        self.residual = ResidualBlock(8, 8) # Residual block with 8 input and output channels
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu_conv2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.4) # Dropout layer with a probability of 0.5
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu_conv1(self.conv1(x))
        x = self.residual(x)
        x = self.relu_conv2(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 16 * 14 * 14)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
