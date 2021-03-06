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
demo_type = int(sys.argv[2])
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']
data_path = config['data_path']

use_dir = config['use_dir'] if 'use_dir' in config else True
z_dim = config['z_dim'] if 'z_dim' in config else 1024

render_near = config['render_near'] if 'render_near' in config else 0.5
render_far = config['render_far'] if 'render_far' in config else 1.5

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

if demo_type == 0:
    save_demo(generator, './demo.png', 8, 8)
elif demo_type == 1:
    n_pose = 8
    poses = [[0.2 * np.cos(2 * np.pi * i / n_pose), 0.2 * np.sin(2 * np.pi * i / n_pose)] for i in range(n_pose)]
    demo_multiview(generator, './demo_multiview.png', poses, 8)
elif demo_type == 2:
    n_pose = 9
    poses = [[0.15 * (i - (n_pose - 1) / 2), 0] for i in range(n_pose)]
    demo_multiview(generator, './demo_extrapolate.png', poses, 8)
elif demo_type == 3:
    n_pose = 5
    poses = [[0, 0, 6 + 6 * i] for i in range(n_pose)]
    demo_multiview(generator, './demo_fov.png', poses, 8)
elif demo_type == 4:
    # Render
    poses = [[angle, 0] for angle in np.linspace(-1, 1, 40 + 1)[:-1]]
    demo_video(generator, './demo.gif', poses)
elif demo_type == 5:
    demo_interpolate(generator, './demo_interpolate.png', 9)
elif demo_type == 6:
    demo_style_mix(generator, './demo_style_mix.png', 8)

