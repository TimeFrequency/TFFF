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
