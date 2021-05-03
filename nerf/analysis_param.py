import json
import os
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np
from data_loader import *


def show_similarity(x, data):
    psnr = [np.mean(np.mean(d['psnr']['in'])) for d in data]
    ssim = [np.mean(np.mean(d['ssim']['in'])) for d in data]
    lpips = [np.mean(np.mean(d['lpips']['in'])) for d in data]

    print(psnr)
    print(ssim)
    print(lpips)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(x, psnr, c='r', marker='o', label='psnr')
    lns2 = ax2.plot(x, ssim, c='g', marker='o', label='ssim')
    lns3 = ax2.plot(x, lpips, c='b', marker='o', label='lpips')
    ax1.set_ylim([5, 35])
    ax2.set_ylim([-0.25, 1.25])
    ax1.grid()
    ax1.set_xlabel('Training Set Noise (log10)')
    ax1.set_ylabel('PSNR')
    ax2.set_ylabel('SSIM & LPIPS')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=6)

    plt.xlim([-11, -1])
    plt.subplots_adjust(0.2, 0.2, 0.8, 0.8)
    plt.gcf().set_size_inches(4, 3)
    plt.title('Similarity-Noise Diagram')
    plt.savefig(f'./logs/{logs[0]}/param.png', dpi=600)
    plt.show()

x = [-20, -8]
logs = ['lego', 'lego_siren']
data = []

for log in logs:
    with open(f'./logs/{log}/test.json', 'r') as data_file:
        data.append(json.load(data_file))


show_similarity(x, data)
