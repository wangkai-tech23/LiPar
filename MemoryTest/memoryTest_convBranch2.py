from torch import nn
import torch
from torchsummary import summary


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # group如果等于输入特征模型的深度，则为DW卷积
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class ParDW(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(ParDW, self).__init__()

        # self.branch1 = nn.Sequential(
        #     ConvBNReLU(3, 64, kernel_size=1),
        #     ConvBNReLU(64, 64, stride=8, groups=64)
        # )

        self.branch2 = nn.Sequential(
            ConvBNReLU(3, 128, kernel_size=1),
            ConvBNReLU(128, 128, stride=4, groups=128),
            ConvBN(128, 256, kernel_size=1),
            ConvBNReLU(256, 256, stride=2, groups=256)
        )
        #
        # self.branch3 = nn.Sequential(
        #     ConvBNReLU(3, 32, kernel_size=1),
        #     ConvBNReLU(32, 32,
        #                stride=2, groups=32),
        #     ConvBN(32, 96, kernel_size=1),
        #     ConvBNReLU(96, 96, stride=2, groups=96),
        #     ConvBN(96, 192, kernel_size=1),
        #     ConvBNReLU(192, 192, stride=2, groups=192)
        # )

        # self.conv = ConvBN(512, 512)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        # branch3 = self.branch3(x)
        #
        # outputs = [branch1, branch2, branch3]
        # x = torch.cat(outputs, 1)  # 在channel纬度上进行合并，1是channel的下标
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # x = self.fc(x)
        return branch2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ConvBN(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # group如果等于输入特征模型的深度，则为DW卷积
        padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel)
        )

if __name__ == '__main__':
    num_classes = 5
    net = ParDW(num_classes=num_classes)
    summary(net, (3, 9, 9))
