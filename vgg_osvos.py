import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math
from copy import deepcopy
from custom_layers import center_crop
import torch.nn.modules as modules
import numpy as np


class OSVOS(nn.Module):
    def __init__(self, pretrained=1):
        super(OSVOS, self).__init__()
        lay_list = [[64, 64],
                    ['M', 128, 128],
                    ['M', 256, 256, 256],
                    ['M', 512, 512, 512],
                    ['M', 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]

        print("Constructing OSVOS architecture..")
        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()

        # Construct the network
        for i in range(0, len(lay_list)):
            # Make the layers of the stages
            stages.append(make_layers_osvos(lay_list[i], in_channels[i]))

            # Attention, side_prep and score_dsn start from layer 2
            if i > 0:
                # Make the layers of the preparation step
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))

                # Make the layers of the score_dsn step
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                # upscale.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear'))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        print("Initializing weights..")
        self._initialize_weights(pretrained)

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        conv_res = []
        conv_side = []
        conv_side_up = []
        conv_side_up_cr = []
        x = self.stages[0](x)
        conv_res.append(x)
        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            conv_res.append(x)
            side_temp = self.side_prep[i - 1](x)
            conv_side.append(side_temp)
            up_tmp = self.upscale[i - 1](side_temp)
            conv_side_up.append(up_tmp)
            crop_tmp = center_crop(up_tmp, crop_h, crop_w)
            side.append(crop_tmp)
            conv_side_up_cr.append(crop_tmp)
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))

        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out, conv_res, conv_side, conv_side_up, conv_side_up_cr

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        if pretrained:
            vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                             'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            _vgg = VGG(make_layers(vgg_structure))

            # Load the weights from saved model
            _vgg.load_state_dict(torch.load('../models/vgg16-397923af.pth',
                                            map_location=lambda storage, loc: storage))

            # Load the weights directly from the web
            # _vgg.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))

            inds = find_conv_layers(_vgg)
            k = 0
            for i in range(len(self.stages)):
                for j in range(len(self.stages[i])):
                    if isinstance(self.stages[i][j], nn.Conv2d):
                        self.stages[i][j].weight = deepcopy(_vgg.features[inds[k]].weight)
                        self.stages[i][j].bias = deepcopy(_vgg.features[inds[k]].bias)
                        k += 1


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data


def find_conv_layers(_vgg):
    inds = []
    for i in range(len(_vgg.features)):
        if isinstance(_vgg.features[i], nn.Conv2d):
            inds.append(i)
    return inds


def make_layers_osvos(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
