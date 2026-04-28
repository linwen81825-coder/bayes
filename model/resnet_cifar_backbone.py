import torch
from torch import nn


def make_group_norm(num_channels, max_groups=8):
    # Keep this local to avoid a future circular import if HybridSwitchTransformer
    # imports this backbone as its stem.
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlockGN(nn.Module):
    """ResNet basic block using GroupNorm for federated CIFAR training."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockGN, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = make_group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.norm2 = make_group_norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                make_group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ResNetCIFARBackbone(nn.Module):
    """CIFAR-style ResNet-18 backbone that returns a 4D feature map."""

    def __init__(
        self,
        layers=(2, 2, 2, 2),
        base_channels=64,
        use_layer4=False,
        in_channels=3,
    ):
        super(ResNetCIFARBackbone, self).__init__()
        if len(layers) != 4:
            raise ValueError("ResNetCIFARBackbone expects a 4-stage layer config")

        self.use_layer4 = use_layer4
        self.inplanes = base_channels

        self.conv1 = conv3x3(in_channels, base_channels, stride=1)
        self.norm1 = make_group_norm(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        if use_layer4:
            self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)
            self.out_channels = base_channels * 8
        else:
            self.layer4 = None
            self.out_channels = base_channels * 4

        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(BasicBlockGN(self.inplanes, out_channels, stride=block_stride))
            self.inplanes = out_channels * BasicBlockGN.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x
