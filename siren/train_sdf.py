import os
import sys
import json
import torch
import scipy.io
import numpy as np
from tqdm import tqdm, trange
from module import sdf_model
from utils_sdf import *

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

"""=============== LOAD DATA ==============="""
point_cloud = scipy.io.loadmat('./data/point_cloud/110f6dbf0e6216e9f9a63e9a8c332e52.mat')['p']
point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float, device='cuda', requires_grad=True)

"""=============== START ==============="""
# Model
model = sdf_model(model_type)
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
log_data = {'loss': []}
batch_idx = 0
epoch_idx = 0
for global_step in trange(global_step + 1, iterations + 1):
    on_data, off_point = load_batch(point_cloud_tensor, batch_idx, batch_size)
    batch_idx += 1
    on_point = on_data[:, :3]
    on_norm = on_data[:, -3:]
    if batch_idx * batch_size >= point_cloud_tensor.shape[0]:
        batch_idx = 0
        epoch_idx += 1
        shuffle_idx = torch.randperm(point_cloud_tensor.shape[0])
        batch_pos_rgb = point_cloud_tensor[shuffle_idx]

    on_pred = model(on_point)
    on_grad = torch.autograd.grad(on_pred, [on_point], grad_outputs=torch.ones_like(on_pred), create_graph=True)[0]
    off_pred = model(off_point)
    off_grad = torch.autograd.grad(off_pred, [off_point], grad_outputs=torch.ones_like(off_pred), create_graph=True)[0]
    loss = sdf_loss(on_pred, on_grad, on_norm, off_pred, off_grad)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_data['loss'].append(loss.item())

    if global_step % i_print == 0:
        tqdm.write(f"[Train] Iter: {global_step}({epoch_idx}-{batch_idx}) Loss: {loss.item()}")
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

