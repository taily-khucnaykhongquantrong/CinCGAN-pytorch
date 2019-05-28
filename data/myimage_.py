import os
import os.path

# import random
# import math
# import errno

from data import common

# import numpy as np
import imageio

# import torch
import torch.utils.data as data

# from torchvision import transforms


class MyImage_(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = "MyImage_"
        self.scale = args.scale
        self.idx_scale = 0
        self.benchmark = False
        datapath = args.testpath

        self.filelist = []
        self.imnamelist = []
        if not train:
            for f in os.listdir(datapath):
                filePath = os.path.join(datapath, f)
                self.filelist.append(filePath)
                self.imnamelist.append(f)

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename_hr = os.path.join(
            self.args.testpath[:-2],
            self.imnamelist[idx],
        )
        filename, _ = os.path.splitext(filename)

        hr = imageio.imread(filename_hr)
        hr = common.set_channel([hr], self.args.n_colors)[0]

        lr = imageio.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return (
            common.np2Tensor([lr], self.args.rgb_range)[0],
            common.np2Tensor([hr], self.args.rgb_range)[0],
            filename,
        )

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
