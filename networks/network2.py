import torch.nn as nn
import torch.nn.functional as F
from torch import flatten, rand

class Net(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 24, 8, stride=1) #32 
        self.bn0 = nn.BatchNorm2d(24)
        self.pool = nn.AvgPool2d((3, 3), stride=1)
        self.conv1 = nn.Conv2d(24, 30, 6) #32
        self.bn1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 36, 6) #24
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 40, 3) #16, 4
        self.bn3 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(40, 48, 3) #4
        self.bn4 = nn.BatchNorm2d(48)
        #fc
        self.dropout = nn.Dropout(0.7) #0.5
        self.fc1 = nn.Linear(54760, 384, bias=True)
        self.ln1 = nn.LayerNorm(384)
        self.fc2 = nn.Linear(384, 128, bias=True)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_of_classes, bias=False)

    def forward(self, input):
        #block1
        x = self.conv0(input)
        x = self.pool(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = flatten(x, 1)
        #fully-connected
        x = F.relu( self.dropout( self.ln1(self.fc1(x) )) )
        x = F.relu( self.dropout( self.ln2(self.fc2(x) )) )
        x = self.fc3(x)
        return x