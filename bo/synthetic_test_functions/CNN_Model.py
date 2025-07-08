import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerCNN(nn.Module):
    def __init__(
            self, out_channels1=32, out_channels2=64, dropout_prob=0.5, num_classes=10):
        super(TwoLayerCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels1, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(out_channels2 * 8 * 8, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x