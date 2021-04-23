import torch
import os
import imageio
from render import *
from nerf import NeRF
from data_loader import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
output_path = './logs/'
experiment_name = 'lego_1'
check_point_idx = 200000
data_path = '../../nerf-pytorch/data/nerf_synthetic/lego'

render_height = 800
render_width = 800
render_focal = 800
render_near = 2.0
render_far = 6.0
render_coarse_sample_num = 64
render_fine_sample_num = 128

use_fine_model = True

"""=============== START ==============="""

# Model
coarse_model = NeRF()
fine_model = NeRF() if use_fine_model else coarse_model

# Load log directory
log_path = os.path.join(output_path, experiment_name)
check_point_path = os.path.join(log_path, '{:06d}.tar'.format(check_point_idx))
print('Loading from', check_point_path)
check_point = torch.load(check_point_path)
global_step = check_point['global_step']
coarse_model.load_state_dict(check_point['coarse_model'])
if use_fine_model:
    fine_model.load_state_dict(check_point['fine_model'])

# Render
poses = [camera_pos_to_transform_matrix(4.0, angle, -30.0) for angle in np.linspace(-180, 180, 40+1)[:-1]]
with torch.no_grad():
    video = render_video(render_width, render_height, render_focal, poses,
                         render_near, render_far,
                         coarse_model, fine_model,
                         render_coarse_sample_num, render_fine_sample_num
                         )
print('Done, saving', video.shape)
video_path = os.path.join(log_path, 'spiral_{:06d}_rgb.gif'.format(global_step))
imageio.mimwrite(video_path, to8b(video), duration=0.1)
print('Saved to', video_path)
