# -*- coding:utf-8 -*-
import os
import argparse
import shutil
import random
import lpips
import warnings
import numpy as np
import pytorch_ssim

import torch
import torch.nn as nn

from Dataset import get_data_loaders
from basic_unet import UNet
from init_weights import init_weights
from util import set_requires_grad
from log_function import print_options, print_network
from patchGAN_discriminator import NLayerDiscriminator

warnings.filterwarnings("ignore")

""" set flags / seeds """

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def psnr(img1, img2):

    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)

    if mse < 1e-10:
        return 100
    psnr1 = 20 * torch.log10(255 / torch.sqrt(mse))
    return psnr1


class Pix2PixModel(nn.Module):
    def __init__(self, opt, device="cpu"):
        super(Pix2PixModel, self).__init__()
        self.opt = opt
        self.netG = UNet(
            n_in=7,
            n_out=1,
            first_channels=32,
            n_dps=5,
            use_pool=True,
            use_bilinear=True,
            norm_type=opt.norm_G,
            device=device,
        )
        init_weights(self.netG, init_type=opt.init_type)
        self.loss_fn_vgg = lpips.LPIPS(net="vgg")
        if self.opt.isTrain:
            self.netD = NLayerDiscriminator(input_nc=2, norm_D=opt.norm_D, n_layers_D=4)
            init_weights(self.netD, init_type=opt.init_type)
            # define loss functions
            self.criterionGAN = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.MSELoss()
            self.criterionCos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def criterion_D(self, fake_B, real_B, real_A, tlabel):
        fake_in = fake_B
        idf, id_fake = self.netD(torch.cat([real_A, fake_in.detach()], dim=1))

        real_in = real_B
        idr, id_real = self.netD(torch.cat([real_A, real_in], dim=1))

        loss_D = 0.5 * (
            self.criterionGAN(
                id_real,
                torch.cat(
                    [
                        torch.full_like(idr, (tlabel.item() + 1) / 3.0),
                        torch.zeros_like(idr),
                    ],
                    dim=1,
                ),
            )
            + self.criterionGAN(
                id_fake,
                torch.cat(
                    [
                        torch.zeros_like(idf),
                        torch.full_like(idf, (tlabel.item() + 1) / 3.0),
                    ],
                    dim=1,
                ),
            )
        )
        return [loss_D]

    def criterion_G(self, fake_B, real_B, real_A, tlabel, p5, fp5_1, fp5_2):
        fake_in = fake_B
        idfr, id_fake_is_real = self.netD(torch.cat([real_A, fake_in], dim=1))
        loss_G_GAN = self.criterionGAN(
            id_fake_is_real,
            torch.cat(
                [
                    torch.full_like(idfr, (tlabel.item() + 1) / 3.0),
                    torch.zeros_like(idfr),
                ],
                dim=1,
            ),
        )

        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_L1

        loss_fea_cos = (
            -(
                self.criterionCos(p5, fp5_1).squeeze()
                + self.criterionCos(p5, fp5_2).squeeze()
                + self.criterionCos(fp5_1, fp5_2).squeeze()
            ).squeeze()
            / 3
        )

        loss_G = loss_G_GAN + loss_G_L1 + 0.1 * loss_fea_cos

        return [loss_G, loss_G_GAN, loss_G_L1]


if __name__ == "__main__":
    """Hpyer parameters"""
    parser = argparse.ArgumentParser(description="MT-DENet")
    parser.add_argument("--experiment_name", type=str, default="DME_1")
    # training option
    parser.add_argument("--isTrain", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=70000)
    parser.add_argument("--lr_g", type=float, default=0.0001)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--lambda_L1", type=float, default=100.0)
    parser.add_argument("--eval_iters", type=int, default=2000)
    parser.add_argument("--save_iters", type=int, default=2000)
    # model option
    parser.add_argument("--input_nc", type=int, default=7)
    parser.add_argument("--output_nc", type=int, default=1)
    parser.add_argument(
        "--gan_mode", type=str, default="ls", help="(ls|original|hinge)"
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="normal",
        help="[normal|xavier|kaiming|orthogonal]",
    )
    parser.add_argument("--norm_G", type=str, default="instance")
    parser.add_argument("--norm_D", type=str, default="instance")

    # data option
    parser.add_argument("--data_path", type=str, default="/home/Data/DME")
    parser.add_argument("--train_list", type=str, default="./dataset/train_1.txt")
    parser.add_argument("--test_list", type=str, default="./dataset/test_1.txt")
    parser.add_argument("--result_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=0)
    opt = parser.parse_args()

    opt.result_dir = os.path.join(opt.result_dir, opt.experiment_name)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
        print_options(parser, opt)
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(opt.result_dir, os.path.basename(__file__)),
        )
    else:
        print("result_dir exists: ", opt.result_dir)
        "exit()"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    """ device configuration """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """ datasets and dataloader """
    train_loader = get_data_loaders(opt, "train")
    test_loader = get_data_loaders(opt, "test")
    print(
        "train_dataset len:", len(train_loader), "test_dataset len:", len(test_loader)
    )

    """ instantiate network and loss function"""
    model = Pix2PixModel(opt, device)
    print_network(model, opt)

    """ optimizer and scheduler """
    optimizer_G = torch.optim.Adam(
        model.netG.parameters(), lr=opt.lr_g, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        model.netD.parameters(), lr=opt.lr_d, betas=(0.5, 0.999)
    )

    """ training part """
    global_iter = 0
    model = model.to(device)
    model.train()
    while global_iter < opt.num_iters:
        for _, sampled_batch_train in enumerate(train_loader):
            real_A, real_B, flags = (
                sampled_batch_train["images"],
                sampled_batch_train["labels"],
                sampled_batch_train["flag"],
            )
            real_A1, real_A2, real_A3, real_A4 = real_A
            real_A1, real_A2, real_A3, real_A4 = (
                real_A1.to(device),
                real_A2.to(device),
                real_A3.to(device),
                real_A4.to(device),
            )
            real_B = real_B.to(device)

            label, flabel1, flabel2 = flags
            label, flabel1, flabel2 = (
                label.to(device),
                flabel1.to(device),
                flabel2.to(device),
            )

            real_A1_mid = real_A1[:, 3, :, :].unsqueeze(1)
            real_A2_mid = real_A2[:, 3, :, :].unsqueeze(1)
            real_A3_mid = real_A3[:, 3, :, :].unsqueeze(1)
            real_A4_mid = real_A4[:, 3, :, :].unsqueeze(1)
            real_B_mid = real_B[:, 3, :, :].unsqueeze(1)

            p5, fp5_1, fp5_2, fake_B = model.netG(
                real_A1, real_A2, real_A3, real_A4, label, flabel1, flabel2
            )

            # update D
            set_requires_grad(model.netD, True)
            losses_D = model.criterion_D(fake_B, real_B_mid, real_A4_mid, label)
            optimizer_D.zero_grad()
            losses_D[0].backward()
            optimizer_D.step()

            # update G
            set_requires_grad(model.netD, False)
            losses_G = model.criterion_G(
                fake_B, real_B_mid, real_A4_mid, label, p5, fp5_1, fp5_2
            )
            optimizer_G.zero_grad()
            losses_G[0].backward()
            optimizer_G.step()

            if global_iter % 10 == 0:
                print(
                    "loss_D:{}, loss_G:{}, loss_G_GAN_D:{}, loss_l1:{}".format(
                        losses_D[0], losses_G[0], losses_G[1], losses_G[2]
                    )
                )

            global_iter += 1

            if global_iter % opt.eval_iters == 0:

                train_lpips = model.loss_fn_vgg(fake_B, real_B_mid)
                train_paras = "lp_%.3f" % train_lpips

                # model eval
                test_lpips = 0
                test_ssim = 0
                test_psnr = 0
                p = 0
                model.eval()
                with torch.no_grad():
                    for _, sampled_batch_test in enumerate(test_loader):
                        real_A, real_B, flags = (
                            sampled_batch_test["images"],
                            sampled_batch_test["labels"],
                            sampled_batch_test["flag"],
                        )
                        real_A1, real_A2, real_A3, real_A4 = real_A
                        real_A1, real_A2, real_A3, real_A4 = (
                            real_A1.to(device),
                            real_A2.to(device),
                            real_A3.to(device),
                            real_A4.to(device),
                        )

                        real_B = real_B.to(device)
                        real_B_mid = real_B[:, 3, :, :].unsqueeze(1)

                        label, flabel1, flabel2 = flags
                        label, flabel1, flabel2 = (
                            label.to(device),
                            flabel1.to(device),
                            flabel2.to(device),
                        )

                        _, _, _, fake_B = model.netG(
                            real_A1, real_A2, real_A3, real_A4, label, flabel1, flabel2
                        )

                        test_lpips += model.loss_fn_vgg(fake_B, real_B_mid)

                        test_ssim += pytorch_ssim.ssim(fake_B, real_B_mid)

                        fake_B_ = fake_B.detach().squeeze(0) * 255
                        fake_B_ = torch.clip(fake_B_, 0, 255)
                        real_B_mid_ = real_B_mid.squeeze(0) * 255
                        test_psnr += psnr(fake_B_, real_B_mid_)
                        p = p + 1
                model.train()

                # write testing data
                test_lpips /= p
                test_ssim /= p
                test_psnr /= p

                eval_paras = (
                    "ssim_%.6f_" % test_ssim
                    + "lp_%.6f_" % test_lpips
                    + "psnr_%.6f_" % test_psnr
                )
                # model saving
                save_dir = opt.result_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if global_iter % opt.save_iters == 0:
                    state = {"netG": model.netG.state_dict()}
                    torch.save(
                        state,
                        os.path.join(
                            opt.result_dir,
                            "model_G_" + str(global_iter) + eval_paras + ".ckpt",
                        ),
                    )

                # print log
                print(
                    "iter: %d" % global_iter
                    + ", train: "
                    + train_paras
                    + ", test: "
                    + eval_paras
                )

            if global_iter == opt.num_iters:
                break
