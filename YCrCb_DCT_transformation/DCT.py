import cv2
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F


class DCTRecombination(nn.Module):
    def __init__(self, N=8, in_channal=3, num_channels=192):
        super(DCTRecombination, self).__init__()

        self.num_channels = num_channels  # 192 or 64
        self.N = N  # N usually takes 8 in JPEG
        self.fre_len = N * N  # block N * N
        self.in_channal = in_channal
        self.out_channal = N * N * in_channal

        self.dct_conv_restructure = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)
        self.dct_conv_restructure.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)
        self.dct_conv_restructure.weight.requires_grad = False

    def forward(self, x):
        # Input x as YCrCb image
        dct = self.dct_conv_restructure(x)
        dct_y = dct[:, 0:self.fre_len, :, :]
        dct_cr = dct[:, self.fre_len:self.fre_len * 2, :, :]
        dct_cb = dct[:, self.fre_len * 2:self.fre_len * 3, :, :]

        # 64 channels, Y:F=0~21, Cr:F=0~20, Cb:F=0~20
        dct_y = dct_y[:, 0:22, :, :]
        dct_cr = dct_cr[:, 0:21, :, :]
        dct_cb = dct_cb[:, 0:21, :, :]
        dct_64channels = torch.cat((dct_y, dct_cr, dct_cb), dim=1)

        # dct_y_afterchoose = dct_y[:, 0:self.choose_frequency+1, :, :]
        # dct_cr_afterchoose = dct_cr[:, 0:self.choose_frequency + 1, :, :]
        # dct_cb_afterchoose = dct_cb[:, 0:self.choose_frequency + 1, :, :]
        # dct_afterchoose = torch.cat((dct_y_afterchoose, dct_cr_afterchoose, dct_cb_afterchoose), dim=1)

        if self.num_channels == 192:
            return dct
        if self.num_channels == 64:
            return dct_64channels

    def mk_coff(self, N=4, rearrange=True):
        dct_weight = np.zeros((N * N, N, N))
        for k in range(N * N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N=N)  # from low frequency to high frequency
        return out_weight  # transformation matrix

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N=8):
        array_size = N * N
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy()
        for k in range(array_size - 1):
            if (i == 0 or i == N - 1) and j % 2 == 0:
                j += 1
            elif (j == 0 or j == N - 1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k + 1, ...] = src_weight[index, ...]
        return rearrange_weigth


def rgb2ycrcb(img_tensor):
    'RGB to YCrCb'
    img_numpy = img_tensor.numpy()
    img_rgb = img_numpy.transpose(1, 2, 0)
    img_rgb = np.array(img_rgb, dtype='uint8')
    # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    img_ycrcb = np.float32(img_ycrcb).transpose(2, 0, 1)
    img_ycrcb_tensor = torch.from_numpy(img_ycrcb)
    return img_ycrcb_tensor


if __name__ == '__main__':
    pass
