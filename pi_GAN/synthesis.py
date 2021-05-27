import torch
import numpy as np
import os
import sys
import json
from tqdm import tqdm, trange
import imageio
from PIL import Image
from render import *
from modules import *
from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = sys.argv[1]
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']
data_path = config['data_path']

use_dir = config['use_dir'] if 'use_dir' in config else True
z_dim = config['z_dim'] if 'z_dim' in config else 1024

render_near = config['render_near'] if 'render_near' in config else 0.5
render_far = config['render_far'] if 'render_far' in config else 1.5
render_coarse_sample_num = config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 12
render_fine_sample_num = config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 24

resolution = 64
render_coarse_sample_num = 8
render_coarse_sample_num = 16
syn_experiment_name = experiment_name + '_syn'
syn_data = './data/syn_2.png'
iterations = 5000
i_print = 10
i_save = 1000
i_image = 100

"""=============== START ==============="""
# Load Data
sym_image = Image.open(syn_data)
sym_image = sym_image.resize((resolution, resolution), Image.ANTIALIAS)
sym_image = torch.tensor(np.array(sym_image, dtype=np.float32)[..., :3] / 255, device='cuda')

# Model
generator = Generator(z_dim, resolution, render_near, render_far, 12, render_coarse_sample_num, render_fine_sample_num, 0.45, 0.15, use_dir)
discriminator = Discriminator(resolution)
requires_grad(generator, False)
requires_grad(discriminator, False)

# Load log directory
log_path = os.path.join(output_path, experiment_name)
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)
    generator.load_state_dict(check_point['generator'])
    discriminator.load_state_dict(check_point['discriminator'])

syn_log_path = os.path.join(output_path, syn_experiment_name)

# Variable
os.makedirs(syn_log_path, exist_ok=True)
syn_check_points = [os.path.join(syn_log_path, f) for f in sorted(os.listdir(syn_log_path)) if 'tar' in f]
print('Found check_points', syn_check_points)
if len(syn_check_points) > 0:
    syn_check_point_path = syn_check_points[-1]
    print('Synthesis Reloading from', syn_check_point_path)
    syn_check_point = torch.load(syn_check_point_path)
    global_step = syn_check_point['global_step']
    loss_log = syn_check_point['loss_log']
    film_params = syn_check_point['film_params']
else:
    global_step = 0
    loss_log = []
    z = torch.randn(1, generator.input_dim, device='cuda')
    film_params = generator.get_mapping(z)
    film_params = torch.tensor(film_params[0], device='cuda', requires_grad=True)
optimizer = torch.optim.Adam(params=[film_params], lr=1e-4)
global_step += 1

# Training
start = global_step
for global_step in trange(start, iterations + 1):

    # Reconstruct loss
    generator.set_film_params(film_params)
    image = generator.render(0, 0)
    rec_loss = torch.mean((image - sym_image) ** 2)

    # Discriminator loss
    gen_image = []
    for i in range(1):
        gen_image.append(generator.render())
    gen_image = torch.stack(gen_image)
    gen_image = gen_image.permute(0, 3, 1, 2).contiguous()
    gen_label = discriminator(gen_image)
    g_loss = -torch.mean(loss_f(-gen_label))

    optimizer.zero_grad()
    loss = 1e2 * rec_loss + g_loss
    loss.backward()
    optimizer.step()
    loss_log.append(loss.item())

    # Logging
    if global_step % i_print == 0:
        tqdm.write(f"[Train] Iter: {global_step} loss: {loss.item()}")

    if global_step % i_save == 0:
        path = os.path.join(output_path, syn_experiment_name, '{:06d}.tar'.format(global_step))
        torch.save({
            'global_step': global_step,
            'loss_log': loss_log,
            'film_params': film_params,
        }, path)
        tqdm.write(f'Saved checkpoints at {path}')

    if global_step % i_image == 0:
        # Turn on testing mode
        filename = os.path.join(syn_log_path, '{:06d}.png'.format(global_step))
        n_pose = 9
        poses = [[0.15 * (i - (n_pose - 1) / 2), 0] for i in range(n_pose)]
        demo_multiview(generator, filename, poses, film_params=film_params.unsqueeze(0))

generator.renderer.set_params(width=128, height=128, coarse_samples=32, fine_samples=64)
film_params = film_params.unsqueeze(0)

n_pose = 9
poses = [[0.15 * (i - (n_pose - 1) / 2), 0] for i in range(n_pose)]
demo_multiview(generator, os.path.join(syn_log_path, 'demo.png'), poses, film_params=film_params)

poses = [[angle, 0] for angle in np.linspace(-1, 1, 40 + 1)[:-1]]
demo_video(generator, os.path.join(syn_log_path, 'demo.gif'), poses, film_params=film_params)

