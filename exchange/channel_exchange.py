import cv2
import torch.nn as nn
import torch
import numpy as np
import math


class NewExchange(nn.Module):
    def __init__(self):
        super(NewExchange, self).__init__()

    def forward(self, features, bns, bn_threshold):
        # features[0] is time domain features, feature[1] is frequency domain features
        bn1, bn2 = bns[0].weight.abs(), bns[1].weight.abs()

        feature1, feature2 = torch.zeros_like(features[0]), torch.zeros_like(features[1])
        bn1_idx_big2small = bn1.sort(0, True)[1].cpu().numpy().tolist()
        bn2_idx_big2small = bn2.sort(0, True)[1].cpu().numpy().tolist()
        # Keep important features in time domain
        feature1[:, bn1 >= bn_threshold] = features[0][:, bn1 >= bn_threshold]

        # Exchange unimportant features in the time domain with important features in the frequency domain
        exchange_list1 = []
        for idx1, value1 in enumerate((bn1 < bn_threshold).cpu().numpy().tolist()):
            if value1:
                exchange_list1.append(idx1)
        if len(exchange_list1) != 0:
            for index1, channel1 in enumerate(exchange_list1):
                feature1[:, channel1, :, :] = features[1][:, bn2_idx_big2small[index1], :, :]

        # Keep important features in frequency domain
        feature2[:, bn2 >= bn_threshold] = features[1][:, bn2 >= bn_threshold]

        # Exchange unimportant features in the frequency domain with important features in the time domain
        exchange_list2 = []
        for idx2, value2 in enumerate((bn2 < bn_threshold).cpu().numpy().tolist()):
            if value2:
                exchange_list2.append(idx2)

        if len(exchange_list2) != 0:
            for index2, channel2 in enumerate(exchange_list2):
                feature2[:, channel2, :, :] = features[0][:, bn1_idx_big2small[index2], :, :]
        return [feature1, feature2]
