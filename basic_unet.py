""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from normlizations import get_norm_layer
import dgl
from dgl.nn.pytorch.conv import GATConv
import sys


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, first_channels, n_dps, use_pool, norm_layer):
        super(UnetEncoder, self).__init__()
        self.inc = InConv(in_channels, first_channels, use_pool, norm_layer)
        self.down_blocks = nn.ModuleList()
        in_channels = first_channels
        out_channels = in_channels * 2
        for i in range(n_dps):
            self.down_blocks.append(
                Down(in_channels, out_channels, use_pool, norm_layer)
            )
            in_channels = out_channels
            out_channels = in_channels * 2

    def forward(self, x):
        x = self.inc(x)
        out_features = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            out_features.append(x)
        return out_features


class UnetDecoder(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer):
        super(UnetDecoder, self).__init__()

        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(
                Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer)
            )
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc = nn.Conv2d(out_channels * 2, n_classes, kernel_size=1)

    def forward(self, features):
        pos_feat = len(features) - 1
        x = features[pos_feat]
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
        x = self.outc(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool, norm_layer):
        super().__init__()
        self.double_conv = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool, norm_layer):
        super().__init__()
        if use_pool:
            self.down_conv = nn.Sequential(
                nn.MaxPool2d(2),
                norm_layer(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                ),
                nn.ReLU(inplace=True),
                norm_layer(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                ),
                nn.ReLU(inplace=True),
                SeBlock(out_channels, out_channels),
            )
        else:
            self.down_conv = nn.Sequential(
                norm_layer(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=2, padding=1
                    )
                ),
                nn.ReLU(inplace=True),
                norm_layer(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                ),
                nn.ReLU(inplace=True),
                SeBlock(out_channels, out_channels),
            )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear, norm_layer):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                T_channels, T_channels, kernel_size=2, stride=2
            )
        self.conv = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            norm_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GcnPred(nn.Module):
    def __init__(self, nodes_num=14 * 8, feats_dim=1024, device="cpu"):
        super(GcnPred, self).__init__()

        self.nodes_num = nodes_num
        self.feats_dim = feats_dim
        self.n_hidden = feats_dim
        self.num_heads = 5
        self.device = device

        self._initgraph_()

        self.uPred = nn.ModuleList()
        self.uPred.append(
            GATConv(
                self.feats_dim,
                self.n_hidden,
                num_heads=self.num_heads,
                residual=True,
                activation=F.relu,
            )
        )
        self.uPred.append(
            GATConv(
                self.n_hidden,
                self.n_hidden,
                num_heads=self.num_heads,
                residual=True,
                activation=F.relu,
            )
        )
        self.uPred.append(
            GATConv(
                self.n_hidden,
                self.feats_dim,
                num_heads=self.num_heads,
                residual=True,
                activation=None,
            )
        )

    def forward(self, x, condition=None):

        x_size = x.size()

        nodes_feats = x.view(x_size[0], x_size[1], -1)
        nodes_feats = nodes_feats.transpose(2, 1)
        h = nodes_feats[0, :, :]
        if condition is not None:
            h = h + condition[0, :, :]

        for _, layer in enumerate(self.uPred):
            h = layer(self.g, h)
            h = torch.mean(h, 1)

        y = h.reshape(1, h.size()[0], h.size()[1])
        y = y.transpose(2, 1)
        y = y.view(x_size[0], x_size[1], x_size[2], x_size[3])

        return y

    def _initgraph_(self):
        print("graph init...")
        self.g = dgl.DGLGraph()

        # add nodes
        self.g.add_nodes(self.nodes_num)

        # add edges use edge broadcasting
        in_indx_sou = list(range(self.nodes_num))
        in_indx_tar = list(range(self.nodes_num))
        for u in in_indx_sou:
            if u % 10 == 0:
                # print('u {}'.format(u))
                sys.stdout.flush()
            for v in in_indx_tar:
                self.g.add_edges(u, v)
        # g = dgl.add_self_loop(g)

        self.g = self.g.to(self.device)


class SeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, reduce=16):
        super(SeBlock, self).__init__()
        self.padding = (k_size - 1) // 2

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, stride, self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, k_size, stride, self.padding),
            nn.BatchNorm2d(out_channels),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduce, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_shortcut = x.clone()
        x_shortcut = self.shortcut(x_shortcut)
        y = x_shortcut * self.se(x_shortcut)
        return y + x
