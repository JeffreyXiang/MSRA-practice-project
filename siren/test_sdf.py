import torch
import os
import imageio
import numpy as np
from matplotlib import pyplot as plt

log_path = '.\logs'
log_exp = ['siren_sdf_2', 'relu_sdf_2', 'relu_pe_sdf_2']
log_label = ['SIREN', 'ReLU', 'ReLU P.E.']
log_img = {exp: [] for exp in log_exp}
log_loss = {exp: None for exp in log_exp}

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
            log_loss[exp] = np.load(f, allow_pickle=True).item()['loss']
    log_img[exp] = np.concatenate(log_img[exp], 1)
    plt.plot(log_loss[exp], label=label)

demo_img = np.concatenate([log_img[exp] for exp in log_exp], 0)
imageio.imwrite('./logs/sdf_demo.png', demo_img)

plt.title('Loss-Iters Diagram')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.ylim([-10, 110])
plt.grid()
plt.legend()
plt.savefig('./logs/sdf_figure.png', dpi=600)
plt.show()

