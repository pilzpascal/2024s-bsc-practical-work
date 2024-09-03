import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # images are 5x5 with 16 channels
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, use_dropout=True):                 # x.shape = (b, 32, 32), where b is batch size
        c1 = F.relu(self.conv1(x))                          # c1.shape = (b, 6, 28, 28)
        p2 = self.pool(c1)                                  # p2.shape = (b, 6, 12, 12)
        c3 = F.relu(self.conv2(p2))                         # c3.shape = (b, 16, 10, 10)
        p4 = self.pool(c3)                                  # p4.shape = (b, 16, 5, 5)
        p5 = torch.flatten(p4, 1)                           # p5.shape = (b, 16*5*5) = (b, 400)
        d6 = F.dropout(p5, p=0.25, training=use_dropout)    # d6.shape = (b, 400)
        f7 = F.relu(self.fc1(d6))                           # f6.shape = (b, 120)
        d8 = F.dropout(f7, p=0.5, training=use_dropout)     # d8.shape = (b, 12)
        f9 = F.relu(self.fc2(d8))                           # f9.shape = (b, 84)
        output = self.fc3(f9)                               # output.shape = (b, 10)
        return output
