import logging
import argparse
import torch
import os
import random
import config
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datasets.aircraft import AircraftDataset
from datasets.bird import BirdsDataset
from datasets.car import CarsDataset
from models.TF_fusion_model import UpScaleResnet50
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'



