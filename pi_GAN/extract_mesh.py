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

render_near = config['render_near'] if 'render_near' in config else 0.5
render_far = config['render_far'] if 'render_far' in config else 1.5
render_coarse_sample_num = config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 12
render_fine_sample_num = config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 24

resolution = 128
render_coarse_sample_num = 32
render_coarse_sample_num = 64

"""=============== START ==============="""
# Model
generator = Generator(z_dim, resolution, render_near, render_far, 12, render_coarse_sample_num, render_fine_sample_num, 0.3, 0.15, use_dir)

# Load log directory
log_path = os.path.join(output_path, experiment_name)
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)
    generator.load_state_dict(check_point['generator'])

create_mesh(generator, 'test', N=512, max_batch=65536)

