import os
import random
import numpy as np
from torchvision import transforms
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from YCrCb_DCT_transformation.DCT import DCT, rgb2ycrcb
from config import root_of_aircraft
from datasets._init_ import mean_train_dct_192_air, mean_test_dct_192_air, std_train_dct_192_air, std_test_dct_192_air, \
    mean_train_dct_64_air, mean_test_dct_64_air, std_train_dct_64_air, std_test_dct_64_air


