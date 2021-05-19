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

render_near = config['render_near'] if 'render_near' in config else 0.5
render_far = config['render_far'] if 'render_far' in config else 1.5
render_coarse_sample_num = config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 12
render_fine_sample_num = config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 24

z_dim = 1024
resolution = 32

"""=============== START ==============="""
# Load Dataset
log_path = os.path.join(output_path, experiment_name)
os.makedirs(log_path, exist_ok=True)
dataset = DataLoader(data_path, 1, resize=resolution/64, preload=False)

# Model
generator = Generator(z_dim, resolution, render_near, render_far, 12, render_coarse_sample_num, render_fine_sample_num, 0.3, 0.15)
discriminator = Discriminator(resolution)

# Load log directory
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)
    loss_log = check_point['loss_log']
    generator.load_state_dict(check_point['generator'])
    discriminator.load_state_dict(check_point['discriminator'])


save_demo(generator, './demo.png')
