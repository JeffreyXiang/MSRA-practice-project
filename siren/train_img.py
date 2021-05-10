import os
import torch
import numpy as np
from tqdm import tqdm, trange
import imageio
from matplotlib import pyplot as plt
from module import SirenMLP, ReLUMLP
from test_img import render_image, to8b

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
output_path = './logs/'
experiment_name = 'ReLU'

iterations = 100000
batch_size = 8192
learning_rate = 1e-3
learning_rate_decay = 500

i_print = 100
i_save = 10000
i_image = 1000

"""=============== LOAD DATA ==============="""
image = imageio.imread('./data/image/lenna.jpg')
image = image.astype(float) / 255
image = image[..., :1]
height, width = image.shape[:2]

rgb = image.reshape((-1, 1))
pos = np.meshgrid(np.array(list(range(width))) / width, np.array(list(range(height))) / height)
pos = np.concatenate([pos[0].reshape((-1, 1)), pos[1].reshape((-1, 1))], axis=1)
pos_rgb = np.concatenate([pos, rgb], axis=1)
np.random.shuffle(pos_rgb)
pos_rgb_tensor = torch.tensor(pos_rgb, dtype=torch.float, device='cuda')


"""=============== START ==============="""
# Model
model = ReLUMLP(2, 1, 256, 3)
trainable_variables = list(model.parameters())
optimizer = torch.optim.Adam(params=trainable_variables, lr=learning_rate, betas=(0.9, 0.999))

# Load log directory
log_path = os.path.join(output_path, experiment_name)
os.makedirs(log_path, exist_ok=True)
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)

    global_step = check_point['global_step']
    optimizer.load_state_dict(check_point['optimizer'])
    model.load_state_dict(check_point['model'])
else:
    global_step = 0

# Trainning
batch_idx = 0
epoch_idx = 0
for global_step in trange(global_step + 1, iterations + 1):
    batch_pos_rgb = pos_rgb_tensor[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    if (batch_idx + 1) * batch_size >= pos_rgb.shape[0]:
        batch_idx = 0
        epoch_idx += 1
        np.random.shuffle(pos_rgb)
        pos_rgb_tensor = torch.tensor(pos_rgb, dtype=torch.float, device='cuda')
    batch_idx += 1
    batch_pos = batch_pos_rgb[:, :2]
    batch_rgb = batch_pos_rgb[:, -1:]

    rgb_pred = model(batch_pos)
    loss = torch.mean((batch_rgb - rgb_pred)**2)
    psnr = -10 * torch.log10(loss)
    loss.backward()
    optimizer.step()

    decay_rate = 0.1
    decay_steps = learning_rate_decay * 1000
    new_learning_rate = learning_rate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate

    if global_step % i_print == 0:
        tqdm.write(f"[Train] Iter: {global_step}({epoch_idx}-{batch_idx}) Loss: {loss.item()} PSNR: {psnr.item()}")
    if global_step % i_image == 0:
        show_image = render_image(model, width, height)
        rgb8 = to8b(show_image)
        filename = os.path.join(log_path, '{:06d}.png'.format(global_step))
        imageio.imwrite(filename, rgb8)
    if global_step % i_save == 0:
        path = os.path.join(output_path, experiment_name, '{:06d}.tar'.format(global_step))
        torch.save({
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)
        tqdm.write(f'Saved checkpoints at {path}')

