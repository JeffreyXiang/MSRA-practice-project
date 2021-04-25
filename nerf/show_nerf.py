import torch
import os
import sys
import imageio
from render import *
from nerf import NeRF
from data_loader import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = os.path.join(sys.argv[1], 'config.json')
check_point_idx = int(sys.argv[2])
render_width = int(sys.argv[3]) if len(sys.argv) > 3 else 400
render_height = int(sys.argv[4]) if len(sys.argv) > 4 else 400
render_focal = int(sys.argv[5]) if len(sys.argv) > 5 else render_width * 1.3875
render_more_sample = float(sys.argv[6]) if len(sys.argv) > 6 else 1
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']

render_near = config['render_near'] if 'render_near' in config else 2.0
render_far = config['render_far'] if 'render_far' in config else 6.0
render_coarse_sample_num = render_more_sample * config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 64
render_fine_sample_num = render_more_sample * config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 128

use_fine_model = config['use_fine_model'] if 'use_fine_model' in config else True

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
print('Done, saving', video[0].shape)
video_rgb_path = os.path.join(log_path, 'spiral_{:06d}_rgb.gif'.format(global_step))
video_alpha_path = os.path.join(log_path, 'spiral_{:06d}_alpha.gif'.format(global_step))
imageio.mimwrite(video_rgb_path, to8b(video[0]), duration=0.1)
print('Saved to', video_rgb_path)
imageio.mimwrite(video_alpha_path, to8b(video[2]), duration=0.1)
print('Saved to', video_alpha_path)
