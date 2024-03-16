# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import torch
from torch import nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-7, m=None):
        super(AngularPenaltySMLoss, self).__init__()

        self.m = 4. if not m else m

        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)

        numerator = torch.cos(self.m * torch.acos(
            torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl), dim=1)
        L = numerator - torch.log(denominator)

        return -torch.mean(L)

"""
class SphereCNN(nn.Module):
    def __init__(self, class_num: int, feature=False):
        super(SphereCNN, self).__init__()
        self.class_num = class_num
        self.feature = feature

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.bnorm4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.dropout = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(8192, 512)

        self.angular = AngularPenaltySMLoss(512, self.class_num)

    def forward(self, x, y):
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bnorm4(self.conv4(x)))  # batch_size (0) * out_channels (1) * height (2) * width (3)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # batch_size (0) * (out_channels * height * width)
        x = F.relu(self.fc5(x))
        #x = self.dropout(x)

        if self.feature:
            return x
        else:
            x_angle = self.angular(x, y)
            return x, x_angle
"""

class SphereCNN(nn.Module):
    def __init__(self, class_num: int, feature=False):
        super(SphereCNN, self).__init__()
        self.class_num = class_num
        self.feature = feature

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv13 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.PReLU(32)
        self.conv21 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.PReLU(128)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.PReLU(256)
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.PReLU(512)

        #self.fc5 = nn.Linear(512 * 5 * 5, 512)
        #self.fc5 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(2048, 512)
        self.angular = AngularPenaltySMLoss(512, self.class_num)

    def forward(self, x, y):
        x1 = self.relu1(self.conv11(x))
        x = self.relu1(self.conv12(x1))
        x = self.relu1(self.conv13(x))
        x = x1 + F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x2 = self.relu2(self.conv21(x))
        x = self.relu2(self.conv22(x2))
        x = self.relu2(self.conv23(x))
        x = x2 + F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x3 = self.relu3(self.conv31(x))
        x = self.relu3(self.conv32(x3))
        x = self.relu3(self.conv33(x))
        x = x3 + F.interpolate(x, size=x3.shape[2:], mode='nearest')
        x4 = self.relu4(self.conv41(x))
        x = self.relu4(self.conv42(x4))
        x = self.relu4(self.conv43(x))
        #x = x4 + F.interpolate(x, size=x4.shape[2:], mode='nearest')

        # batch_size (0) * out_channels (1) * height (2) * width (3)

        x = x.view(x.size(0), -1)  # batch_size (0) * (out_channels * height * width)
        x = self.fc5(x)

        if self.feature:
            return x
        else:
            x_angle = self.angular(x, y)
            return x, x_angle


if __name__ == "__main__":
    net = SphereCNN(50)
    input = torch.ones(64, 3, 96, 96)
    output = net(input, None)
