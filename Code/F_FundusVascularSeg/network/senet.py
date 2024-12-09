from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

pretrained_settings = {
    'senet154'           : 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
    'se_resnet50'        : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'se_resnet101'       : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
    'se_resnet152'       : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
    'se_resnext50_32x4d' : 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
}

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1  = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.fc2  = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)

    def forward(self, x):
        weight = x.mean(dim=(2,3), keepdim=True)
        weight = F.relu(self.fc1(weight), inplace=True)
        weight = torch.sigmoid(self.fc2(weight))
        return weight*x

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1( x )), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(self.se_module(out) + x, inplace=True)

class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes * 2)
        self.conv2      = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2        = nn.BatchNorm2d(planes * 4)
        self.conv3      = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.se_module  = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride     = stride

class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width           = math.floor(planes * (base_width/64)) * groups
        self.conv1      = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1        = nn.BatchNorm2d(width)
        self.conv2      = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2        = nn.BatchNorm2d(width)
        self.conv3      = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.se_module  = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride     = stride

class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, snapshot, inplanes=128, 
                 input_3x3=True, downsample_kernel_size=3, downsample_padding=1):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.snapshot = snapshot
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1',   nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2',   nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3',   nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1',   nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer0( x )
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load(self.snapshot), strict=False)

def senet154():
    return SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16)

def se_resnext50_32x4d():
    return SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, snapshot='../pretrain/se_resnext50_32x4d-a260b3a4.pth', inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0)

def se_resnext101_32x4d():
    return SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16, inplanes=64,  input_3x3=False, downsample_kernel_size=1, downsample_padding=0)
