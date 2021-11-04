from posixpath import join
import torch
import numpy as np
import os
import time

from PIL import Image
from network import VGG
from torchvision import transforms
from random import sample
from ssim import SSIM
import matplotlib.pyplot as plt
import cv2

plt.rcParams["font.family"] = "Times New Roman"

filenames = [
    [
        '/data/lxd/datasets/contents_original/Ann_Hathaway_P00045.jpg',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Ann_Hathaway_P00045/Ann_Hathaway_P00045-5.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Ann_Hathaway_P00045/Ann_Hathaway_P00045-18.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Ann_Hathaway_P00045/Ann_Hathaway_P00045-46.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Ann Hathaway/Ann_Hathaway_P00045_0.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Ann Hathaway/Ann_Hathaway_P00045_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Ann Hathaway/Ann_Hathaway_P00045_4.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Ann Hathaway/1910_Ann Hathaway_P00045_C00025_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Ann Hathaway/1861_Ann Hathaway_P00045_C00019_2.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Ann Hathaway/1906_Ann Hathaway_P00045_C00044_1.png',
    ],
    [
        '/data/lxd/datasets/contents_original/Lily Collins_P00009.jpg',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Lily_Collins_P00009/Lily_Collins_P00009-5.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Lily_Collins_P00009/Lily_Collins_P00009-21.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Lily_Collins_P00009/Lily_Collins_P00009-39.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Lily Collins/Lily_Collins_P00009_0.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Lily Collins/Lily_Collins_P00009_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Lily Collins/Lily_Collins_P00009_4.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Lily Collins/1806_Lily Collins_P00009_C00001_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Lily Collins/1807_Lily Collins_P00009_C00006_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Lily Collins/1810_Lily Collins_P00009_C00002_1.png',
    ],
    [
        '/data/lxd/datasets/contents_original/Donald_Sutherland_P00015.jpg',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Donald_Sutherland_P00015/Donald_Sutherland_P00015-95.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Donald_Sutherland_P00015/Donald_Sutherland_P00015-47.png',
        '/data/lxd/datasets/UserStudy/2021-05-16-CAST/Donald_Sutherland_P00015/Donald_Sutherland_P00015-17.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Donald Sutherland/Donald_Sutherland_P00015_0.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Donald Sutherland/Donald_Sutherland_P00015_3.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-WarpGAN/Donald Sutherland/Donald_Sutherland_P00015_4.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Donald Sutherland/1535_Donald Sutherland_P00015_C00010_2.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Donald Sutherland/1538_Donald Sutherland_P00015_C00008_1.png',
        '/data/lxd/datasets/UserStudy/2021-10-04-CariGAN/Donald Sutherland/1543_Donald Sutherland_P00015_C00015_1.png',
    ]
]
output_path = 'output/2021-10-20-diversity-select'
os.makedirs(output_path, exist_ok=True)

plot = []

for row in filenames:
    row_img = []
    for filename in row:
        row_img.append(cv2.imread(filename))
    plot.append(np.hstack(row_img))

plot = np.vstack(plot)
cv2.imwrite(os.path.join(output_path, 'plot.png'), plot)
