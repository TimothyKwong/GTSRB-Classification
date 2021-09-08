import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, rand

class Net(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 16, 6, stride=1)
        self.pool0 = nn.MaxPool2d((6, 6), stride=1)
        self.bn0 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 24, 6)
        self.pool1 = nn.MaxPool2d((5, 5), stride=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, 6)
        self.pool2 = nn.MaxPool2d((4, 4), stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 40, 6)
        self.pool3 = nn.MaxPool2d((3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(40, 48, 3)
        self.pool4 = nn.MaxPool2d((3, 3), stride=1)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(48, 56, 3)
        self.pool5 = nn.MaxPool2d((3, 3), stride=1)
        self.bn5 = nn.BatchNorm2d(56)
        self.conv6 = nn.Conv2d(56, 64, 3)
        self.pool6 = nn.MaxPool2d((3, 3), stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 72, 3)
        self.pool7 = nn.MaxPool2d((3, 3), stride=1)
        self.bn7 = nn.BatchNorm2d(27)
        #fc
        #40368 - Overfitted
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(32448, 1024, bias=True)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, num_of_classes, bias=False)

    def forward(self, input):
        x = self.conv0(input)
        x = self.pool0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = flatten(x, 1)
        #fully-connected
        x = F.relu( self.dropout( self.ln1(self.fc1(x) )) )
        x = self.fc2(x) #Note: No dropout in final layer, as stated in Hinton (2012)
                        #Note: No activation in final layer since crossentropyloss
        return x
