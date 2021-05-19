import json
import os
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np
from data_loader import *

def b_spline(x, y, s=1):
    x_new = np.linspace(min(x), max(x), 1024)
    x_idx = sorted(enumerate(x), key=lambda x: x[1])
    x = [x[i[0]] for i in x_idx]
    y = [y[i[0]] for i in x_idx]
    f = interpolate.UnivariateSpline(x, y, k=5, s=s)
    y_bspline = f(x_new)
    return x_new, y_bspline

def show_similarity(data, label, data2=None, label2=None, show_train=False):
    plt.subplots_adjust(0.1, 0.1, 0.9, 0.9, hspace=0.5)
    plt.gcf().set_size_inches(6, 8)

    plt.subplot(3, 1, 1)
    if show_train:
        plt.scatter(data['dist']['train'], data['psnr']['train'], c='m', marker='o', s=5, label='train')
    plt.scatter(data['dist']['in'], data['psnr']['in'], c='g', marker='s', s=5)
    plt.scatter(data['dist']['ex'], data['psnr']['ex'], c='b', marker='s', s=5)
    x = data['dist']['in'] + data['dist']['ex']
    y = data['psnr']['in'] + data['psnr']['ex']
    x, y = b_spline(x, y, 500)
    plt.plot(x, y, c='c', label=label)
    if data2 is not None:
        plt.scatter(data2['dist']['in'], data2['psnr']['in'], c='r', marker='s', s=5)
        plt.scatter(data2['dist']['ex'], data2['psnr']['ex'], c='y', marker='s', s=5)
        x = data2['dist']['in'] + data2['dist']['ex']
        y = data2['psnr']['in'] + data2['psnr']['ex']
        x, y = b_spline(x, y, 300)
        plt.plot(x, y, c='orange', label=label2)
    plt.axis([0, 180, 0, 50])
    plt.grid()
    plt.legend()
    plt.title('PSNR-Distance Diagram')
    plt.xlabel('Angle(°)')

    plt.subplot(3, 1, 2)
    if show_train:
        plt.scatter(data['dist']['train'], data['ssim']['train'], c='m', marker='o', s=5, label='train')
    plt.scatter(data['dist']['in'], data['ssim']['in'], c='g', marker='s', s=5)
    plt.scatter(data['dist']['ex'], data['ssim']['ex'], c='b', marker='s', s=5)
    x = data['dist']['in'] + data['dist']['ex']
    y = data['ssim']['in'] + data['ssim']['ex']
    x, y = b_spline(x, y)
    plt.plot(x, y, c='c', label=label)
    if data2 is not None:
        plt.scatter(data2['dist']['in'], data2['ssim']['in'], c='r', marker='s', s=5)
        plt.scatter(data2['dist']['ex'], data2['ssim']['ex'], c='y', marker='s', s=5)
        x = data2['dist']['in'] + data2['dist']['ex']
        y = data2['ssim']['in'] + data2['ssim']['ex']
        x, y = b_spline(x, y)
        plt.plot(x, y, c='orange', label=label2)
    plt.axis([0, 180, 0.2, 1.1])
    plt.grid()
    plt.legend()
    plt.title('SSIM-Distance Diagram')
    plt.xlabel('Angle(°)')

    plt.subplot(3, 1, 3)
    if show_train:
        plt.scatter(data['dist']['train'], data['lpips']['train'], c='m', marker='o', s=5, label='train')
    plt.scatter(data['dist']['in'], data['lpips']['in'], c='g', marker='s', s=5)
    plt.scatter(data['dist']['ex'], data['lpips']['ex'], c='b', marker='s', s=5)
    x = data['dist']['in'] + data['dist']['ex']
    y = data['lpips']['in'] + data['lpips']['ex']
    x, y = b_spline(x, y)
    plt.plot(x, y, c='c', label=label)
    if data2 is not None:
        plt.scatter(data2['dist']['in'], data2['lpips']['in'], c='r', marker='s', s=5)
        plt.scatter(data2['dist']['ex'], data2['lpips']['ex'], c='y', marker='s', s=5)
        x = data2['dist']['in'] + data2['dist']['ex']
        y = data2['lpips']['in'] + data2['lpips']['ex']
        x, y = b_spline(x, y)
        plt.plot(x, y, c='orange', label=label2)
    plt.axis([0, 180, -0.1, 0.8])
    plt.grid()
    plt.legend()
    plt.title('LPIPS-Distance Diagram')
    plt.xlabel('Angle(°)')

    plt.savefig(f'./logs/{log1}/similarity.png', dpi=600)
    plt.show()

log1 = 'lego_range_3_30'
log2 = log1 + '_alpha'
label1 = 'no_alpha'
label2 = 'with_alpha'
data1 = None
data2 = None

with open(f'./logs/{log1}/config.json', 'r') as config_file:
    config = json.load(config_file)

data_path = config['data_path']
data_resize = config['data_resize'] if 'data_resize' in config else 0.5
data_skip = config['data_skip'] if 'data_skip' in config else 8
data_train_idx = config['data_train_idx'] if 'data_train_idx' in config else None
data_view_dir_range = config['data_view_dir_range'] if 'data_view_dir_range' in config else None

# Load Dataset
_, poses, _, _, _, _ = load_blender_data(data_path, data_resize, data_skip, data_view_dir_range, None, data_train_idx)
show_data_distribution(poses, save_name=log1)

with open(f'./logs/{log1}/test.json', 'r') as data_file:
    data1 = json.load(data_file)
if log2 is not None:
    with open(f'./logs/{log2}/test.json', 'r') as data_file:
        data2 = json.load(data_file)

show_similarity(data1, label1, data2, label2, show_train=True)
print(np.mean(data1['psnr']['train']), np.mean(data1['psnr']['in']), np.mean(data1['psnr']['ex']))
print(np.mean(data1['ssim']['train']), np.mean(data1['ssim']['in']), np.mean(data1['ssim']['ex']))
print(np.mean(data1['lpips']['train']), np.mean(data1['lpips']['in']), np.mean(data1['lpips']['ex']))
