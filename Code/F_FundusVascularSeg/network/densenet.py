import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple

class _DenseLayer(nn.Module):
    def __init__( self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    # @torch.jit.unused  # noqa: T484
    # def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
    #     def closure(*inputs):
    #         return self.bn_function(inputs)
    #     return cp.checkpoint(closure, *input)

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input: List[Tensor]) -> Tensor:
    #     pass

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input: Tensor) -> Tensor:
    #     pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer( num_input_features + i*growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_init_features, bn_size=4, drop_rate=0, num_classes=1000, snapshot=None):
        super(DenseNet, self).__init__()
        self.snapshot = snapshot
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out      = F.relu(features, inplace=True).mean(dim=(2,3))
        out      = self.classifier(out)
        return out

    def initialize(self):
        pattern  = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        snapshot = torch.load(self.snapshot)

        for key in list(snapshot.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                snapshot[new_key] = snapshot[key]
                del snapshot[key]
            self.load_state_dict(snapshot)


def DenseNet121():
    return DenseNet(32, (6, 12, 24, 16), 64, snapshot='../pretrain/densenet121-a639ec97.pth')

def DenseNet161():
    return DenseNet(48, (6, 12, 36, 24), 96, snapshot='../pretrain/densenet161-8d451a50.pth')

def DenseNet169():
    return DenseNet(32, (6, 12, 32, 32), 64, snapshot='../pretrain/densenet169-b2777c0a.pth')

def DenseNet201():
    return DenseNet(32, (6, 12, 48, 32), 64, snapshot='../pretrain/densenet201-c1103571.pth')