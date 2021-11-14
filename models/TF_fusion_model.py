import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from exchange.channel_exchange import NewExchange
import cv2


# resnet
class BasicBlockFusion(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn_threshold=2e-2):
        super(BasicBlockFusion, self).__init__()
        self.bn_threshold = bn_threshold

        # time dimain branch
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # frequency domain branch
        self.conv1_f = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1_f = nn.BatchNorm2d(planes)
        self.conv2_f = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_f = nn.BatchNorm2d(planes)

        # time domain down sample
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # frequency domain down sample
        self.downsample_f = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample_f = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # the list of bn.weight()
        self.t_f_bn_layer1 = [self.bn1, self.bn1_f]
        self.t_f_bn_layer2 = [self.bn2, self.bn2_f]

        # exchange function
        self.exchange = NewExchange()

    def forward(self, x_t_f_list):

        out_t = self.bn1(self.conv1(x_t_f_list[0]))
        out_f = self.bn1_f(self.conv1_f(x_t_f_list[1]))
        out_t, out_f = self.exchange([out_t, out_f], self.t_f_bn_layer1, self.bn_threshold)  # Local fusion
        out_t, out_f = F.relu(out_t), F.relu(out_f)

        out_t = self.bn2(self.conv2(out_t))
        out_f = self.bn2_f(self.conv2_f(out_f))
        out_t += self.downsample(x_t_f_list[0])
        out_f += self.downsample_f(x_t_f_list[1])

        out_t, out_f = F.relu(out_t), F.relu(out_f)

        return [out_t, out_f]


class BottleneckFusion(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_threshold=2e-2):
        super(BottleneckFusion, self).__init__()
        self.bn_threshold = bn_threshold

        # first layer -- T
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # first layer -- F
        self.conv1_f = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1_f = nn.BatchNorm2d(planes)
        # second layer --T
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # second layer --F
        self.conv2_f = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_f = nn.BatchNorm2d(planes)
        # three layer -- T
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        # three layer -- F
        self.conv3_f = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3_f = nn.BatchNorm2d(self.expansion * planes)
        # down sample -- T
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # down sample -- F
        self.downsample_f = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample_f = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # the list of bn.weight()
        self.t_f_bn_layer1 = [self.bn1, self.bn1_f]
        self.t_f_bn_layer2 = [self.bn2, self.bn2_f]
        self.t_f_bn_layer3 = [self.bn3, self.bn3_f]

        # exchange function
        self.exchange = NewExchange()

    def forward(self, x_t_f_list):
        out_t = F.relu(self.bn1(self.conv1(x_t_f_list[0])))
        out_f = F.relu(self.bn1_f(self.conv1_f(x_t_f_list[1])))

        out_t, out_f = self.bn2(self.conv2(out_t)), self.bn2_f(self.conv2_f(out_f))
        out_t, out_f = self.exchange([out_t, out_f], self.t_f_bn_layer2, self.bn_threshold)  # Local fusion
        out_t, out_f = F.relu(out_t), F.relu(out_f)

        out_t, out_f = self.bn3(self.conv3(out_t)), self.bn3_f(self.conv3_f(out_f))
        out_t += self.downsample(x_t_f_list[0])
        out_f += self.downsample_f(x_t_f_list[1])
        out_t, out_f = F.relu(out_t), F.relu(out_f)
        return [out_t, out_f]


class UpScaleDctResnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bn_threshold=2e-2):
        super(UpScaleDctResnet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1_f = nn.Conv2d(192, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1_f = nn.BatchNorm2d(64)
        # self.bn_first_f = nn.BatchNorm2d(192)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, bn_threshold=bn_threshold)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, bn_threshold=bn_threshold)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, bn_threshold=bn_threshold)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, bn_threshold=bn_threshold)
        self.fc_t = nn.Linear(512 * block.expansion, num_classes)
        self.fc_f = nn.Linear(512 * block.expansion, num_classes)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # training parameters
        self.alpha = nn.Parameter(torch.ones(2, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

        # initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BottleneckFusion):
                nn.init.constant_(m.bn3.weight, 0)
                nn.init.constant_(m.bn3_f.weight, 0)
            elif isinstance(m, BasicBlockFusion):
                nn.init.constant_(m.bn2.weight, 0)
                nn.init.constant_(m.bn2_f.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride, bn_threshold):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_threshold))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, dct_img):
        x_t = x  # time domain input
        x_f = dct_img  # frequency doamin input
        x_f = self.upsample(x_f)  # Upsampling to align time domain

        out_t = self.maxpool(F.relu(self.bn1(self.conv1(x_t))))
        # out_f = self.maxpool(F.relu(self.bn1_f(self.conv1_f(x_f))))
        out_t, out_f = self.layer1([out_t, x_f])
        out_t, out_f = self.layer2([out_t, out_f])
        out_t, out_f = self.layer3([out_t, out_f])
        out_t, out_f = self.layer4([out_t, out_f])
        out_t, out_f = self.avgpool(out_t), self.avgpool(out_f)
        out_t, out_f = out_t.view(x.size(0), -1), out_f.view(x.size(0), -1)
        out_t, out_f = self.fc_t(out_t), self.fc_f(out_f)

        # Gloabl fusion in the decision-making layer
        alpha_beta = F.softmax(self.alpha, dim=0)
        out_t_f = alpha_beta[0] * out_t.detach() + alpha_beta[1] * out_f.detach()
        return [out_t, out_f, out_t_f], alpha_beta


def UpScaleResnet9(num_classes):
    return UpScaleDctResnet(BasicBlockFusion, [1, 1, 1, 1], num_classes=num_classes)

def UpScaleResnet18(num_classes):
    return UpScaleDctResnet(BasicBlockFusion, [2, 2, 2, 2], num_classes=num_classes)

def UpScaleResnet34(num_classes):
    return UpScaleDctResnet(BasicBlockFusion, [3, 4, 6, 3], num_classes=num_classes)

def UpScaleResnet50(num_classes):
    return UpScaleDctResnet(BottleneckFusion, [3, 4, 6, 3], num_classes=num_classes)

def UpScaleResnet101(num_classes):
    return UpScaleDctResnet(BottleneckFusion, [3, 4, 23, 3], num_classes=num_classes)

def UpScaleResnet152(num_classes):
    return UpScaleDctResnet(BottleneckFusion, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    net = UpScaleResnet9(5)
    data1 = torch.randn(1, 3, 448, 448)
    data2 = torch.randn(1, 64, 56, 56)
    outputs, alfa_beta = net(data1, data2)

