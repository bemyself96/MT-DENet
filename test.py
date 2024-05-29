# -*- coding:utf-8 -*-

import argparse
import torch
import os
import glob

# import scipy.io as sio
from torchvision.utils import save_image
from basic_unet import UNet
from Dataset import *

# import cv2
import pytorch_ssim
import lpips

""" set flags / seeds """
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="MT-DENet")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--data_path", type=str, default="/home/Data/DME")
parser.add_argument("--test_list", type=str, default="./dataset/test_1.txt")
parser.add_argument(
    "--model_path",
    type=str,
    default="./logs/DME_1/model_G_70000.ckpt",
)
parser.add_argument("--results_dir", type=str, default="/home/Data/result/DME_1")
opt = parser.parse_args()

results_realA = os.path.join(opt.results_dir, "realA")
if not os.path.exists(results_realA):
    os.makedirs(results_realA)
results_realB = os.path.join(opt.results_dir, "realB")
if not os.path.exists(results_realB):
    os.makedirs(results_realB)

results_fakeB = os.path.join(opt.results_dir, "fakeB")
if not os.path.exists(results_fakeB):
    os.makedirs(results_fakeB)


def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * torch.log10(255 / torch.sqrt(mse))
    return psnr1


if __name__ == "__main__":

    """datasets and dataloader"""
    test_loader = get_data_loaders(opt, "test")

    """ device configuration """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG = UNet(
        n_in=7,
        n_out=1,
        first_channels=32,
        n_dps=5,
        use_pool=True,
        use_bilinear=True,
        norm_type="instance",
        device=device,
    )
    netG = netG.to(device)
    netG.eval()
    loss_LPIPS = lpips.LPIPS(net="vgg")
    loss_LPIPS = loss_LPIPS.to(device)
    loss_LPIPS.eval()

    """models init or load checkpoint"""
    # print(torch.load(opt.model_save_path))
    netG.load_state_dict(torch.load(opt.model_path)["netG"])

    ssim_val = 0
    lpips_val = 0
    psnr_val = 0
    p = 0

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
            real_A_mid = real_A1[:, 3, :, :].unsqueeze(1)

            label, flabel1, flabel2 = flags
            label, flabel1, flabel2 = (
                label.to(device),
                flabel1.to(device),
                flabel2.to(device),
            )

            _, _, _, fake_B = netG(
                real_A1, real_A2, real_A3, real_A4, label, flabel1, flabel2
            )

            filename = sampled_batch_test["filename"]
            filename = filename[0].split(".")[0]

            # save_image(real_A_mid, os.path.join(results_realA, filename + ".bmp"))
            # save_image(real_B_mid, os.path.join(results_realB, filename + ".bmp"))
            save_image(fake_B, os.path.join(results_fakeB, filename + ".bmp"))

            ssim_val += pytorch_ssim.ssim(fake_B, real_B_mid)

            lpips_val += loss_LPIPS(fake_B, real_B_mid)

            fake_B_ = fake_B.detach().squeeze(0) * 255
            fake_B_ = torch.clip(fake_B_, 0, 255)
            real_B_mid_ = real_B_mid.squeeze(0) * 255
            psnr_val += psnr1(fake_B_, real_B_mid_)
            p += 1

            # write testing data
        ssim_val /= p
        lpips_val /= p
        psnr_val /= p
        print(
            "ssim_val:{}, lpips_val:{},psnr_val:{}".format(
                ssim_val, lpips_val, psnr_val
            )
        )
