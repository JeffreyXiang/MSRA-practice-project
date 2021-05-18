import torch
import numpy as np
import imageio

to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)

def summary_module(module):
    print(list(module.named_modules()))

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def loss_f(u):
    return -torch.nn.functional.softplus(-u)

def loss_r1(y, x):
    batch_size = x.shape[0]
    gradients = torch.autograd.grad(y, [x], torch.ones_like(y), create_graph=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    res = torch.mean(gradients.norm(dim=-1) ** 2)
    return res
