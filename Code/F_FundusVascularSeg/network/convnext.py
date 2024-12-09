import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path):
        super().__init__()
        self.dwconv     = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm       = LayerNorm(dim, eps=1e-6)
        self.pwconv1    = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act        = nn.GELU()
        self.pwconv2    = nn.Linear(4 * dim, dim)
        self.gamma      = nn.Parameter(torch.ones((dim)),  requires_grad=True)
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input   = x
        x       = self.dwconv(x)
        x       = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x       = self.norm(x)
        x       = self.pwconv1(x)
        x       = self.act(x)
        x       = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
    Args:
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4, snapshot=None):
        super().__init__()
        self.snapshot          = snapshot
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.downsample_layers.append(nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        self.downsample_layers.append(nn.Sequential(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)))
        self.downsample_layers.append(nn.Sequential(LayerNorm(dims[1], eps=1e-6, data_format="channels_first"), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)))
        self.downsample_layers.append(nn.Sequential(LayerNorm(dims[2], eps=1e-6, data_format="channels_first"), nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates    = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur         = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        x1 = self.downsample_layers[0](x)
        x1 = self.stages[0](x1)

        x2 = self.downsample_layers[1](x1)
        x2 = self.stages[1](x2)

        x3 = self.downsample_layers[2](x2)
        x3 = self.stages[2](x3)

        x4 = self.downsample_layers[3](x3)
        x4 = self.stages[3](x4)
        return x1, x2, x3, x4
    
    def initialize(self):
        self.load_state_dict(torch.load(self.snapshot)['model'], strict=False)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight             = nn.Parameter(torch.ones(normalized_shape))
        self.bias               = nn.Parameter(torch.zeros(normalized_shape))
        self.eps                = eps
        self.data_format        = data_format
        self.normalized_shape   = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def convnext_tiny():
    # model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], snapshot='../pretrain/convnext_tiny_1k_224_ema.pth')
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], snapshot='../pretrain/convnext_tiny_22k_224.pth')
    return model
