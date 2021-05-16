import torch
import os
import imageio
import numpy as np
from matplotlib import pyplot as plt

log_path = '.\logs'
log_exp = ['siren_img', 'siren_img_1', 'tanh_img', 'relu_img', 'relu_pe_img']
log_label = ['SIREN', 'SIREN\'', 'Tanh', 'ReLU', 'ReLU P.E.']
log_img = {exp: [] for exp in log_exp}
log_psnr = {exp: None for exp in log_exp}

for exp, label in zip(log_exp, log_label):
    path = os.path.join(log_path, exp)
    for f in sorted(os.listdir(path)):
        if 'png' in f:
            f = os.path.join(path, f)
            print(f)
            log_img[exp].append(imageio.imread(f))
        if 'npy' in f:
            f = os.path.join(path, f)
            print(f)
            log_psnr[exp] = np.load(f, allow_pickle=True).item()['psnr']
    log_img[exp] = np.concatenate(log_img[exp], 1)
    plt.plot(log_psnr[exp], label=label)

demo_img = np.concatenate([log_img[exp] for exp in log_exp], 0)
imageio.imwrite('./logs/img_demo.png', demo_img)

plt.title('PSNR-Iters Diagram')
plt.xlabel('Iterations')
plt.ylabel('PSNR')
plt.grid()
plt.legend()
plt.savefig('./logs/img_figure.png', dpi=600)
plt.show()

