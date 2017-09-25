import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)


class ConvBlock(nn.Module):
    """Simple Convolutional Block consisting of conv - bn - ReLU layers"""

    def __init__(self, numIn, numOut, stride=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(numIn, numIn, stride)
        self.bn1 = nn.BatchNorm2d(numIn)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample  # TODO: fill the method
        self.stride = stride

        if numIn != numOut:
            self.project = nn.Conv2d(numIn, numOut, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(numOut)
        else:
            self.project = None

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.project is not None:
            out = self.project(out)
            out = self.bn2(out)
            out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    """Residual block without bottleneck"""

    def __init__(self, numIn, numOut, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(numIn, numOut, stride)
        self.bn1 = nn.BatchNorm2d(numOut)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(numOut, numOut)
        self.bn2 = nn.BatchNorm2d(numOut)
        self.downsample = downsample
        self.stride = stride

        self.bn3 = nn.BatchNorm2d(numOut)

        if numIn != numOut:
            self.project = nn.Conv2d(numIn, numOut, kernel_size=1)
        else:
            self.project = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.project is not None:
            residual = self.project(residual)

        residual = self.bn3(residual)

        out += residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    """Residual block with bottleneck, suggested by
    Kaiming He et al., CVPR 2016"""

    def __init__(self, numIn, numOut, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(numIn, numOut/2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(numOut/2)
        self.conv2 = nn.Conv2d(numOut/2, numOut/2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(numOut/2)
        self.conv3 = nn.Conv2d(numOut/2, numOut, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(numOut)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(numOut)
        self.downsample = downsample
        self.stride = stride
        if numIn != numOut:
            self.project = nn.Conv2d(numIn, numOut, kernel_size=1)
        else:
            self.project = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.project is not None:
            residual = self.project(x)

        residual = self.bn4(residual)

        out += residual
        out = self.relu(out)
        return out


class BottleneckPreact(nn.Module):
    """Residual block with bottleneck, improvement suggested by
    Kaiming He et al., ECCV 2016"""

    def __init__(self, numIn, numOut, stride=1, downsample=None):
        super(BottleneckPreact, self).__init__()

        self.bn1 = nn.BatchNorm2d(numIn)
        self.conv1 = nn.Conv2d(numIn, numOut/2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(numOut / 2)
        self.conv2 = nn.Conv2d(numOut/2, numOut/2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(numOut/2)
        self.conv3 = nn.Conv2d(numOut/2, numOut, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(numIn)

        self.downsample = downsample
        self.stride = stride
        if numIn != numOut:
            self.project = nn.Conv2d(numIn, numOut, kernel_size=1)
        else:
            self.project = None

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        residual = self.bn4(residual)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.project is not None:
            residual = self.project(residual)

        out += residual
        return out


class HourGlass(nn.Module):
    """Create a single hourglass"""

    def __init__(self, numHGscales, block, nFeat=32, nModules=1):
        self.inplanes = 64
        super(HourGlass, self).__init__()

        # Upper Branch
        self.up1 = self._make_layer(block, nFeat, nModules)

        # Lower Branch
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.low1 = self._make_layer(block, nFeat, nModules)

        # Create internal hourglass recursively
        if numHGscales > 1:
            self.low2 = HourGlass(numHGscales-1, block, nFeat, nModules)
        else:
            self.low2 = self._make_layer(block, nFeat, nModules)

        self.low3 = self._make_layer(block, nFeat, nModules)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def _make_layer(self, block, nFeat, nModules):
        layers = []
        for i in range(0, nModules):
            layers.append(block(nFeat, nFeat))

        return nn.Sequential(*layers)

    def forward(self, x):
        b_up = self.up1(x)
        b_low = self.up2(self.low3(self.low2(self.low1(self.maxpool(x)))))
        out = b_up + b_low
        return out


class Net_SHG(nn.Module):
    """Create the stacked hourglass network"""

    def __init__(self, nStack, nHGscales, blockstr, nFeat=32, nModules=1):
        self.nStack = nStack
        super(Net_SHG, self).__init__()
        if blockstr == 'ConvBlock':
            block = ConvBlock
        elif blockstr == 'BasicBlock':
            block = BasicBlock
        elif blockstr == 'BottleNeck':
            block = BottleNeck
        elif blockstr == 'BottleneckPreact':
            block = BottleneckPreact

        print('Initializing {} hourglasses with {} blocks'.format(nStack, blockstr))

        self.conv1 = nn.Conv2d(3, nFeat/4, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 128
        self.bn1 = nn.BatchNorm2d(nFeat/4)
        self.relu = nn.ReLU(inplace=True)

        self.r1 = block(nFeat/4, nFeat/2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.r4 = block(nFeat/2, nFeat/2)
        self.r5 = block(nFeat/2, nFeat)

        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')

        hg = []
        convout = []
        for i in range(0, nStack):
            layers = []
            layers.append(HourGlass(nHGscales, block, nFeat, nModules))

            for j in range(0, nModules):
                layers.append(self._make_layer(block, nFeat, nModules))
            convout.append(nn.Conv2d(nFeat, 1, kernel_size=3, padding=1))
            hg.append(nn.Sequential(*layers))

        # Use ListModule to create lists of layers, originally not allowed by PyTorch
        self.hg = ListModule(*hg)
        self.convout = ListModule(*convout)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(.5)
                m.bias.data.zero_()

    def _make_layer(self, block, nFeat, nModules):
        layers = []
        for i in range(0, nModules):
            layers.append(block(nFeat, nFeat))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.r1(x))
        x = self.r5(self.r4(x))

        out = []
        for i in range(0, self.nStack):
            x = self.hg[i](x)
            out.append(self.up4(self.convout[i](x)))

        return out


class ListModule(nn.Module):
    """ Allows the creation of lists of layers,
        which is not supported by PyTorch in general.
        Detailed discussion here: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219
    """
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

