import torch
import os
import sys
import json
from tqdm import tqdm, trange
import imageio
from dataloader import DataLoader
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
iterations = config['iterations'] if 'iterations' in config else [50000]
resolution = config['resolution'] if 'resolution' in config else [32]

render_near = config['render_near'] if 'render_near' in config else 0.5
render_far = config['render_far'] if 'render_far' in config else 1.5
render_coarse_sample_num = config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 12
render_fine_sample_num = config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 24


"""=============== START ==============="""
# Model
generator = Generator(z_dim, resolution[0], render_near, render_far, 12, render_coarse_sample_num, render_fine_sample_num, 0.45, 0.15, use_dir)
discriminator = Discriminator(resolution[0])

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
    loss_log = check_point['loss_log']
    generator.load_state_dict(check_point['generator'])
    discriminator.load_state_dict(check_point['discriminator'])

stage = 0
for i in range(len(iterations)):
    if global_step > iterations[i]:
        stage = i
    else:
        break
dataset = DataLoader(data_path, 1, resize=resolution[stage]/64, preload=False)
generator.set_resolution(resolution[stage])
discriminator.set_resolution(resolution[stage])
print(f'Starting at stage {stage}, resolution:{resolution[stage]}')

print('Real Image:')
for i in range(8):
    _, _, real_image = dataset.get()
    real_image = real_image.permute(0, 3, 1, 2).contiguous()
    real_label = discriminator(real_image)
    print(real_label)

print('Generated Image:')
for i in range(8):
    z = torch.randn(1, z_dim)
    gen_image = generator(z)
    gen_label = discriminator(gen_image)
    print(gen_label)

plt.plot(loss_log['g_loss'], label='g_loss')
plt.plot(loss_log['d_loss'], label='d_loss')
plt.legend()
plt.title('Loss-Iterations Diagram')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('figure.png', dpi=600)
plt.show()
