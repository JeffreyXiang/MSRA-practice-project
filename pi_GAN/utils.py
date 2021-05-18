import torch
import numpy as np
import imageio

to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)

def summary_module(module):
    # print(list(module.named_modules()))
    total_params = sum(p.numel() for p in module.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in module.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

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

@torch.no_grad()
def save_demo(generator, renderer, file_name):
    z = torch.randn(16, generator.input_dim, device='cuda')
    nerf, film_params = generator(z)
    gen_image = []
    for i in range(film_params.shape[0]):
        nerf.set_film_params(film_params[i])
        gen_image.append(renderer.render(nerf).cpu().numpy())
    gen_image_row = [
        np.concatenate(gen_image[0:4], axis=1),
        np.concatenate(gen_image[4:8], axis=1),
        np.concatenate(gen_image[8:12], axis=1),
        np.concatenate(gen_image[12:16], axis=1)
    ]
    demo_image = np.concatenate(gen_image_row, axis=0)
    rgb = to8b(demo_image)
    imageio.imsave(file_name, rgb)

