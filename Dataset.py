# coding:UTF-8
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms


def get_data_loaders(opt, status="train"):

    if status == "test":

        return DataLoader(TaskDataset(opt, "test"))

    if status == "train":

        return DataLoader(
            TaskDataset(opt, "train"),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
        )


def read_path(data_path, file_list):

    imlist = []
    with open(file_list, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(os.path.join(data_path, impath))
    return imlist


class TaskDataset(Dataset):
    def __init__(self, opt, status="train"):

        if status == "train":
            self.datalist = read_path(opt.data_path, opt.train_list)
            print("# train samples: {}".format(len(self.datalist)))

        if status == "test":
            self.datalist = read_path(opt.data_path, opt.test_list)
            print("# test samples: {}".format(len(self.datalist)))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        filename = self.datalist[idx]

        labels = [0, 1, 2]

        if "Monthly" in filename:
            label = 0
        elif "GILA" in filename:
            label = 1
        else:
            label = 2

        labels.remove(label)

        flabel1 = labels[0]
        flabel2 = labels[1]

        data = np.load(filename).astype(np.float32)

        vol_V1 = torch.from_numpy(data[0] / 255)
        vol_V2 = torch.from_numpy(data[1] / 255)
        vol_V3 = torch.from_numpy(data[2] / 255)
        vol_V4 = torch.from_numpy(data[3] / 255)
        vol_V14 = torch.from_numpy(data[4] / 255)

        filename_sp = filename.split("/")
        imgname = filename_sp[-1]

        sample = {
            "images": [vol_V1, vol_V2, vol_V3, vol_V4],
            "labels": vol_V14,
            "flag": [label, flabel1, flabel2],
            "filename": imgname,
        }

        return sample
