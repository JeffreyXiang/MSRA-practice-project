import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm, trange
import imageio
from matplotlib import pyplot as plt
from module import SirenMLP, ReLUMLP
from test_img import render_image, to8b

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = sys.argv[1]
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']

iterations = config['iterations'] if 'iterations' in config else 10000
batch_size = config['batch_size'] if 'batch_size' in config else 65536
learning_rate = config['learning_rate'] if 'learning_rate' in config else 1e-4
model_type = config['model_type'] if 'model_type' in config else 'siren'

i_print = config['i_print'] if 'i_print' in config else 100
i_save = config['i_save'] if 'i_save' in config else 10000
i_image = config['i_image'] if 'i_image' in config else 1000

"""=============== LOAD DATA ==============="""
image = imageio.imread('./data/image/cameraman.jpg')
image = image.astype(float) / 255
image = np.expand_dims(image, 2)
height, width = image.shape[:2]

rgb = image.reshape((-1, 1))
pos = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
pos = np.concatenate([pos[0].reshape((-1, 1)), pos[1].reshape((-1, 1))], axis=1)
pos_rgb = np.concatenate([pos, rgb], axis=1)
np.random.shuffle(pos_rgb)
pos_rgb_tensor = torch.tensor(pos_rgb, dtype=torch.float, device='cuda')


"""=============== START ==============="""
# Model
if model_type == 'siren':
    model = SirenMLP(2, 1, 256, 3)
elif model_type == 'relu':
    model = ReLUMLP(2, 1, 256, 3)
for name, param in model.named_parameters():
    print(name)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

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
log_data = {'loss': [], 'psnr': []}
batch_idx = 0
epoch_idx = 0
for global_step in trange(global_step + 1, iterations + 1):
    batch_pos_rgb = pos_rgb_tensor[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    batch_idx += 1
    batch_pos = batch_pos_rgb[:, :2]
    batch_rgb = batch_pos_rgb[:, -1:]
    if batch_idx * batch_size >= pos_rgb.shape[0]:
        batch_idx = 0
        epoch_idx += 1
        # shuffle_idx = torch.randperm(batch_pos_rgb.shape[0])
        # batch_pos_rgb = batch_pos_rgb[shuffle_idx]

    rgb_pred = model(batch_pos)
    loss = torch.mean((batch_rgb - rgb_pred)**2)
    psnr = -10 * torch.log10(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_data['loss'].append(loss.item())
    log_data['psnr'].append(psnr.item())

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

log_data_path = os.path.join(log_path, 'log.npy')
print(f'log data save to: {log_data_path}')
np.save(log_data_path, log_data)
