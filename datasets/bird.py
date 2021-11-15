import os
import random
import numpy as np
from torchvision import transforms
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import root_of_bird
from YCrCb_DCT_transformation.DCT import DCT, rgb2ycrcb
from datasets._init_ import mean_train_dct_192_bird, mean_test_dct_192_bird, std_train_dct_192_bird, std_test_dct_192_bird, \
    mean_train_dct_64_bird, mean_test_dct_64_bird, std_train_dct_64_bird, std_test_dct_64_bird

# Make label

