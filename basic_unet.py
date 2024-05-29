""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from normlizations import get_norm_layer
import dgl
from dgl.nn.pytorch.conv import GATConv
import sys


class UNet(nn.Module):
    def __init__(
        self,
        n_in=9,
        n_out=1,
        first_channels=64,
        n_dps=4,
        use_bilinear=False,
        use_pool=False,
        norm_type="instance",
        device="cpu",
    ):
        super(UNet, self).__init__()

        self.alpha = 0.9

        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder(n_in, first_channels, n_dps, use_pool, norm_layer)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder(
            n_out, first_channels, n_dps, use_bilinear, norm_layer
        )

        self.gcn1 = GcnPred(nodes_num=14 * 8, feats_dim=1024, device=device)
        self.gcn2 = GcnPred(nodes_num=14 * 8, feats_dim=1024, device=device)
        self.gcn3 = GcnPred(nodes_num=14 * 8, feats_dim=1024, device=device)
        self.gcn4 = GcnPred(nodes_num=14 * 8, feats_dim=1024, device=device)

        self.projhead = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(first_channels, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, bias=False),
        )

        self.emb_layer = nn.Sequential(
            nn.Embedding(3, 14 * 8),
            nn.SiLU(),
            nn.Linear(14 * 8, 14 * 8),
        )

    def forward(self, x1, x2, x3, x4, cond, fcond1, fcond2):

        fe1 = self.encoder(x1)
        fe2 = self.encoder(x2)
        fe3 = self.encoder(x3)
        fe4 = self.encoder(x4)

        fe1[-1] = self.gcn1(fe1[-1])
        fe2[-1] = self.gcn2(self.alpha * fe2[-1] + (1 - self.alpha) * fe1[-1])
        fe3[-1] = self.gcn3(self.alpha * fe3[-1] + (1 - self.alpha) * fe2[-1])

        fcondition1 = self.emb_layer(fcond1)
        fcondition1 = fcondition1[:, :, None].repeat(1, 1, 1024)
        ffe4_1 = self.gcn4(
            self.alpha * fe4[-1] + (1 - self.alpha) * fe3[-1], fcondition1
        )

        fcondition2 = self.emb_layer(fcond2)
        fcondition2 = fcondition2[:, :, None].repeat(1, 1, 1024)
        ffe4_2 = self.gcn4(
            self.alpha * fe4[-1] + (1 - self.alpha) * fe3[-1], fcondition2
        )

        condition = self.emb_layer(cond)
        condition = condition[:, :, None].repeat(1, 1, 1024)
        fe4[-1] = self.gcn4(
            self.alpha * fe4[-1] + (1 - self.alpha) * fe3[-1], condition
        )

        # fe4[-1] = self.gcn4(self.alpha * fe4[-1] + (1 - self.alpha) * fe3[-1])

        out = self.decoder(fe4)

        p4 = self.projhead(fe4[-1])
        fp4_1 = self.projhead(ffe4_1)
        fp4_2 = self.projhead(ffe4_2)

        return p4, fp4_1, fp4_2, out


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
