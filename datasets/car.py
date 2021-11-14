import os
import random
import numpy as np
from torchvision import transforms
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from YCrCb_DCT_transformation.DCT import DCT, rgb2ycrcb
import torch.nn as nn
from PIL import ImageFile
from config import root_of_car
from datasets._init_ import mean_train_dct_192_car, mean_test_dct_192_car, std_train_dct_192_car, std_test_dct_192_car, \
    mean_train_dct_64_car, mean_test_dct_64_car, std_train_dct_64_car, std_test_dct_64_car
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Make label
labels_dict = []
for root, dirs, _ in os.walk(root_of_car + "train"):
    for idx, sub_dir in enumerate(dirs):
        labels_dict.append([sub_dir, idx])

labels_dict = dict(labels_dict)


class CarsDataset(Dataset):
    def __init__(self, data_dir, num_channels, transform_448=None, train=True):
        # self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)
        self.transform_448 = transform_448
        self.DCT = DCT(N=8, in_channal=3, num_channels=num_channels)
        self.train = train
        self.num_channels = num_channels

    def __getitem__(self, index):

        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform_448 is not None:
            img_448 = self.transform_448(img)
        img_448_rgb_after_normal = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_448)

        img_448_rgb = img_448 * 255
        img_448_ycrcb = rgb2ycrcb(img_448_rgb)  # YCrCb transformation
        img_448_ycrcb_tensor = torch.unsqueeze(img_448_ycrcb, dim=0)  # DCT transformation
        dct_img = self.DCT(img_448_ycrcb_tensor).squeeze()
        if self.train:
            if self.num_channels == 192:
                dct_img = transforms.Normalize(mean_train_dct_192_car, std_train_dct_192_car)(dct_img)
            if self.num_channels == 64:
                dct_img = transforms.Normalize(mean_train_dct_64_car, std_train_dct_64_car)(dct_img)
        else:
            if self.num_channels == 192:
                dct_img = transforms.Normalize(mean_test_dct_192_car, std_test_dct_192_car)(dct_img)
            if self.num_channels == 64:
                dct_img = transforms.Normalize(mean_test_dct_64_car, std_test_dct_64_car)(dct_img)

        return img_448_rgb_after_normal, dct_img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = root + '/' + sub_dir + "/" + img_name
                    # path_img = os.path.join(root, sub_dir, img_name)
                    label = labels_dict[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info
