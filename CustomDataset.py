
import os
import random
import copy
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from options import options as opt
import pytorch_lightning as pl
from torchvision import transforms

from utils.image_utils import random_augmentation, crop_img

class CustomDataset(Dataset):
    def __init__(self, degradation_path, gt_path, train):
        self.degradation_path = degradation_path
        self.gt_path = gt_path
        self.train = train  # True or False决定是训练集还是验证集
        self.degradation_image_name_list = os.listdir(self.degradation_path)
        self.gt_image_name_list = os.listdir(self.gt_path)

    def _crop_patch(self, img_1, img_2):  # 统一进行随机裁剪
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - opt.patch_size)
        ind_W = random.randint(0, W - opt.patch_size)

        patch_1 = img_1[ind_H:ind_H + opt.patch_size, ind_W:ind_W + opt.patch_size]
        patch_2 = img_2[ind_H:ind_H + opt.patch_size, ind_W:ind_W + opt.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        degradation_image_name = self.degradation_image_name_list[idx]
        gt_image_name = self.gt_image_name_list[idx]

        degradation_image = np.array(Image.open(os.path.join(self.degradation_path, degradation_image_name)).convert('RGB'))
        gt_image = np.array(Image.open(os.path.join(self.gt_path, gt_image_name)).convert('RGB'))

        # 对称裁剪确保为base的倍数，是必需的，不然后续的pixel_unshuffle等模块会报错
        degradation_image = crop_img(degradation_image, base=16)
        gt_image = crop_img(gt_image, base=16)

        # 根据self.train决定是否应用数据增强
        if self.train:
            degradation_image, gt_image = random_augmentation(*self._crop_patch(degradation_image, gt_image))  # 随机裁剪 + 随机变换
        else:
            degradation_image, gt_image = self._crop_patch(degradation_image, gt_image)

        # 最后ToTensor
        degradation_tensor = transforms.ToTensor()(degradation_image)
        gt_tensor = transforms.ToTensor()(gt_image)

        return degradation_tensor, gt_tensor

    def __len__(self):
        return len(self.degradation_image_name_list)