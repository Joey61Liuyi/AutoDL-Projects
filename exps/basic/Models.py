# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 16:29
# @Author  : LIU YI


import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import _LRScheduler

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=100, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model

        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvNetMaker(nn.Module):
    """
    Creates a simple (plane) convolutional neural network
    """

    def __init__(self, layers):
        """
        Makes a cnn using the provided list of layers specification
        The details of this list is available in the paper
        :param layers: a list of strings, representing layers like ["CB32", "CB32", "FC10"]
        """
        super(ConvNetMaker, self).__init__()
        self.conv_layers = []
        self.fc_layers = []
        h, w, d = 32, 32, 3
        previous_layer_filter_count = 3
        previous_layer_size = h * w * d
        num_fc_layers_remained = len([1 for l in layers if l.startswith('FC')])
        for layer in layers:
            if layer.startswith('Conv'):
                filter_count = int(layer[4:])
                self.conv_layers += [nn.Conv2d(previous_layer_filter_count, filter_count, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(filter_count), nn.ReLU(inplace=True)]
                previous_layer_filter_count = filter_count
                d = filter_count
                previous_layer_size = h * w * d
            elif layer.startswith('MaxPool'):
                self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                h, w = int(h / 2.0), int(w / 2.0)
                previous_layer_size = h * w * d
            elif layer.startswith('FC'):
                num_fc_layers_remained -= 1
                current_layer_size = int(layer[2:])
                if num_fc_layers_remained == 0:
                    self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size)]
                else:
                    self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size), nn.ReLU(inplace=True)]
                previous_layer_size = current_layer_size

        conv_layers = self.conv_layers
        fc_layers = self.fc_layers
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def resnet14_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet8_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
    return model


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet26_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [4, 4, 4], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)

    # model = ResNet(depth=110)
    return model


class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x


def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


def vgg11_cifar(**kwargs):
    model = VGG(get_vgg_layers(vgg11_config, batch_norm=True), class_num)
    return model


def vgg13_cifar(**kwargs):
    model = VGG(get_vgg_layers(vgg13_config, batch_norm=True), class_num)
    return model


def vgg16_cifar(**kwargs):
    model = VGG(get_vgg_layers(vgg16_config, batch_norm=True), class_num)
    return model


def vgg19_cifar(**kwargs):
    model = VGG(get_vgg_layers(vgg19_config, batch_norm=True), kwargs['num_classes'])
    return model


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,
                         stride=2),

            nn.Conv2d(in_channels=64,
                      out_channels=192,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),

            nn.Conv2d(in_channels=192,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # forward: forward propagation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self, number_classes):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv3 = conv3x3(6, 12)
        self.conv2 = nn.Conv2d(12, 16, kernel_size=5)
        self.conv4 = conv3x3(16, 20)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CS_LeNet(nn.Module):
    def __init__(self, opts, class_num):
        super(CS_LeNet, self).__init__()

        self.opts = opts
        self.conv1 = nn.Conv2d(3, self.opts[-1], kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(self.opts[-1], self.opts[-1], kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(self.opts[-1] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.C_parameters = nn.Parameter(torch.randn(2, len(opts)))
        self.tau = 10

    def channel_aggregation(self, hardwts):
        ones = []
        for i in self.opts:
            vec = [1] * i + [0] * (self.opts[-1] - i)
            ones.append(vec)
        ones = torch.Tensor(ones).cuda()
        output = sum(
            ones[_ie] * hardwts[_ie]
            for _ie in range(len(ones))
        )
        return output

    def forward(self, x):
        # idx = 0
        while True:  # 防止生成的alpha加上gumble噪音之后造成崩溃
            # print("we are inside the loop, and this it the loop {}".format(idx))
            # idx = idx+1
            gumbels = -torch.empty_like(self.C_parameters).exponential_().log()  # 当有特征被传进来时，用gumble-softmax把alpha变得连续可导
            logits = (self.C_parameters.log_softmax(dim=1) + gumbels) / self.tau
            probs = F.softmax(logits, dim=1)  # 对Channels的概率分布（对logits进行softmax函数操作）
            index = probs.max(-1, keepdim=True)[1]  # 找到probs中最大值的index
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)  # 根据probs每个元素的最大值的索引生成一个one-hot tensor
            hardwts = one_h - probs.detach() + probs  # ？？？
            if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
            ):
                continue
            else:
                break
        x = self.conv1(x)
        gl = self.channel_aggregation(hardwts[0]).reshape([1, x.shape[1], 1, 1])
        x = torch.mul(gl, x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        gl = self.channel_aggregation(hardwts[1]).reshape([1, x.shape[1], 1, 1])
        x = torch.mul(gl, x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x
    def get_alphas(self):
        return [self.C_parameters]
    def get_weights(self):
        xlist = list(self.conv1.parameters()) + list(self.relu1.parameters()) + list(self.max_pool1.parameters()) \
                + list(self.conv2.parameters()) + list(self.relu2.parameters()) + list(self.max_pool2.parameters())
        xlist += list(self.fc1.parameters()) + list(self.relu3.parameters()) \
                 + list(self.fc2.parameters()) + list(self.relu4.parameters()) + list(self.fc3.parameters())
        return xlist

    def genotype(self):
        logits = (self.C_parameters.log_softmax(dim=1)) / self.tau
        probs = F.softmax(logits, dim=1)  # 对Channels的概率分布（对logits进行softmax函数操作）
        index = probs.max(-1, keepdim=True)[1]  # 找到probs中最大值的index
        one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)  # 根据probs每个元素的最大值的索引生成一个one-hot tensor
        hardwts = one_h - probs.detach() + probs
        layer1 = (self.channel_aggregation(hardwts[0]).view(-1,) == 1).sum()
        layer2 = (self.channel_aggregation(hardwts[1]).view(-1,) == 1).sum()
        return int(layer1), int(layer2)

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.C_parameters, dim=-1).cpu())


class LeNet_wide(nn.Module):
    def __init__(self, class_num, first_channel, second_channel):
        super(LeNet_wide, self).__init__()
        self.conv1 = nn.Conv2d(3, first_channel, kernel_size=5)
        self.conv2 = nn.Conv2d(first_channel, second_channel, kernel_size=5)
        self.fc1 = nn.Linear(second_channel * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)



class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MobileNet_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=100):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNet_Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)  # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16
        self.fire2 = fire(96, 16, 64)
        self.fire3 = fire(128, 16, 64)
        self.fire4 = fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
        self.fire5 = fire(256, 32, 128)
        self.fire6 = fire(256, 48, 192)
        self.fire7 = fire(384, 48, 192)
        self.fire8 = fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
        self.fire9 = fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)
        return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2(num_classes):
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg, num_classes)


def ShuffleNetG3(num_classes):
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups': 3
    }
    return ShuffleNet(cfg, num_classes)

class Block(nn.Module):
    '''expand + depthwise + pointwise + squeeze-excitation'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_planes, out_planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes//16, out_planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(x) if self.stride == 1 else out
        # Squeeze-Excitation
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = out * w + shortcut
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=100):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(cfg[-1][1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def EfficientNetB0(class_num):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 2),
           (6,  24, 2, 1),
           (6,  40, 2, 2),
           (6,  80, 3, 2),
           (6, 112, 3, 1),
           (6, 192, 4, 2),
           (6, 320, 1, 2)]
    return EfficientNet(cfg, class_num)


resnet_book = {
    '8': resnet8_cifar,
    '14': resnet14_cifar,
    '20': resnet20_cifar,
    '26': resnet26_cifar,
    '32': resnet32_cifar,
    '44': resnet44_cifar,
    '56': resnet56_cifar,
    '110': resnet110_cifar,
}
plane_cifar10_book = {
    '2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
    '4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
    '6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
    '8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
          'Conv128', 'Conv128', 'MaxPool', 'FC64', 'FC10'],
    '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
           'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC128', 'FC10'],
}
plane_cifar100_book = {
    '2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
    '4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
    '6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'FC100'],
    '8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
          'Conv256', 'Conv256', 'MaxPool', 'FC64', 'FC100'],
    '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
           'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC512', 'FC100'],
}
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,
                512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M']
vgg_cifar10_book = {
    '11': vgg11_cifar,
    '13': vgg13_cifar,
    '16': vgg16_cifar,
    '19': vgg19_cifar,
}


def is_resnet(name):
    """
    Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
    :param name:
    :return:
    """
    name = name.lower()
    if name.startswith("resnet"):
        return 'resnet'
    elif name.startswith('plane'):
        return 'plane'
    elif name.startswith('alexnet'):
        return 'alexnet'
    elif name.startswith('vgg'):
        return 'vgg'
    elif name.startswith('resnext'):
        return 'resnext'
    elif name.startswith('lenet_wide'):
        return 'lenet_wide'
    elif name.startswith('lenet'):
        return 'lenet'
    elif name.startswith('googlenet'):
        return 'googlenet'
    elif name.startswith('mobilenet'):
        return 'mobilenet'
    elif name.startswith('squeezenet'):
        return 'squeezenet'
    elif name.startswith('shufflenet'):
        return 'shufflenet'
    elif name.startswith('efficientnetb0'):
        return 'efficientnetb0'
    elif name.startswith('cs_lenet'):
        return 'cs_lenet'

def create_cnn_model(name, dataset="cifar100", total_epochs = 160, model_path = None, use_cuda = False):
    """
    Create a student for training, given student name and dataset
    :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
    :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
    :return: a pytorch student for neural network
    """
    num_classes = 100 if dataset == 'cifar100' else 10
    model = None
    scheduler = None

    if is_resnet(name) == 'resnet':
        resnet_size = name[6:]
        resnet_model = resnet_book.get(resnet_size)(num_classes = num_classes)
        model = resnet_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs/2, total_epochs*3/4, total_epochs], gamma=0.1, last_epoch=-1)
        # scheduler = MultiStepLR(optimizer, 5, total_epochs, [total_epochs/2, total_epochs*3/4, total_epochs], 0.1)
    elif is_resnet(name) == 'plane':
        plane_size = name[5:]
        model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(
            plane_size)
        plane_model = ConvNetMaker(model_spec)
        model = plane_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, total_epochs*2, gamma=0.1, last_epoch=-1)

    elif is_resnet(name) == 'vgg':
        vgg_size = name[3:]
        vgg_model = vgg_cifar10_book.get(vgg_size)(num_classes=num_classes)
        model = vgg_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    elif is_resnet(name) == 'alexnet':
        alexnet_model = AlexNet(num_classes)
        model = alexnet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = ExponentialLR(optimizer, 10, total_epochs, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[total_epochs*3/8, total_epochs*3/4, total_epochs], gamma=0.5)

    elif is_resnet(name) == 'lenet':
        lenet_model = LeNet(num_classes)
        model = lenet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    elif is_resnet(name) == 'lenet_wide':

        tep = name.split('_')
        first_channel = int(tep[-2])
        second_channel = int(tep[-1])
        lenet_model = LeNet_wide(num_classes, first_channel, second_channel)
        model = lenet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif is_resnet(name) == "cs_lenet":
        if dataset == 'cifar10':
            channel_range = list(range(4, 24, 2))
        elif dataset == 'cifar100':
            channel_range = list(range(20, 220, 20))
        model = CS_LeNet(channel_range, num_classes)
        w_optimizer = torch.optim.Adam(model.get_weights(), lr=0.001)
        w_scheduler = torch.optim.lr_scheduler.MultiStepLR(w_optimizer,
                                                         [total_epochs * 3 / 8, total_epochs * 3 / 4, total_epochs],
                                                         gamma=0.5)
        a_optimizer = torch.optim.Adam(model.get_alphas(), lr=1e-2, betas=(0.5, 0.999), weight_decay=1e-3)
        a_scheduler = torch.optim.lr_scheduler.MultiStepLR(a_optimizer,
                                                           [total_epochs * 3 / 8, total_epochs * 3 / 4, total_epochs],
                                                           gamma=0.2)

        optimizer = (w_optimizer, a_optimizer)
        scheduler = (w_scheduler, a_scheduler)
    elif is_resnet(name) == 'googlenet':
        googlenet_model = GoogLeNet(num_classes)
        model = googlenet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs / 2, total_epochs * 3 / 4, total_epochs], gamma=0.1, last_epoch=-1)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs*3 / 8, total_epochs * 3 / 4, total_epochs], gamma=0.5)
    elif is_resnet(name) == 'mobilenet':
        mobilenet_model = MobileNet(num_classes)
        model = mobilenet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs*3 / 8, total_epochs * 3 / 4, total_epochs], gamma=0.5)
    elif is_resnet(name) == 'squeezenet':
        squeezenet_model = SqueezeNet(num_classes)
        model = squeezenet_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs / 2, total_epochs * 3 / 4, total_epochs], gamma=0.1, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         [total_epochs * 3 / 8, total_epochs * 3 / 4, total_epochs],
                                                         gamma=0.5)
    elif is_resnet(name) == 'shufflenet':
        shufflenet_type = name[10:]
        if shufflenet_type == 'g2' or shufflenet_type == 'G2':
            shufflenet_model = ShuffleNetG2(num_classes)
        else:
            shufflenet_model = ShuffleNetG3(num_classes)
        model = shufflenet_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [total_epochs*3 / 8, total_epochs * 3 / 4, total_epochs], gamma=0.5)

    elif is_resnet(name) == 'efficientnetb0':
        efficientnetb0_model = EfficientNetB0(num_classes)
        model = efficientnetb0_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        # Assuming optimizer uses lr = 0.05 for all groups
        # lr = 0.05     if epoch < 30
        # lr = 0.005    if 30 <= epoch < 60
        # lr = 0.0005   if 60 <= epoch < 90
        # ...
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         [total_epochs * 3 / 8, total_epochs * 3 / 4, total_epochs],
                                                         gamma=0.5)
    # copy to cuda if activated
    if use_cuda:
        model = model.cuda()

    if model_path:
        print(model_path)
        checkpoint = torch.load(model_path)

        pass

        if 'base-model' in checkpoint:
            model.load_state_dict(checkpoint['base-model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, scheduler


# def load_checkpoint(model, checkpoint_path):
#     """
#     Loads weights from checkpoint
#     :param model: a pytorch nn student
#     :param str checkpoint_path: address/path of a file
#     :return: pytorch nn student with weights loaded from checkpoint
#     """
#     model_ckp = torch.load(checkpoint_path)
#     model.load_state_dict(model_ckp['model_state_dict'])
#     return model


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6