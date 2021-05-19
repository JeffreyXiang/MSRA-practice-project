import torch
import os
import sys
import json
from tqdm import tqdm
import imageio
from data_loader import *
from render import *
from nerf import NeRF, SirenNeRF
import pytorch_ssim
import lpips

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = os.path.join(sys.argv[1], 'config.json')
check_point_idx = int(sys.argv[2])
render_more_sample = float(sys.argv[3]) if len(sys.argv) > 3 else 1

with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']
data_path = config['data_path']
data_resize = config['data_resize'] if 'data_resize' in config else 0.5
data_skip = config['data_skip'] if 'data_skip' in config else 8
data_train_idx = config['data_train_idx'] if 'data_train_idx' in config else None
data_view_dir_range = config['data_view_dir_range'] if 'data_view_dir_range' in config else None
data_show_distribution = False

render_near = config['render_near'] if 'render_near' in config else 2.0
render_far = config['render_far'] if 'render_far' in config else 6.0
render_coarse_sample_num = render_more_sample * (config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 64)
render_fine_sample_num = render_more_sample * (config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 128)

use_fine_model = config['use_fine_model'] if 'use_fine_model' in config else True
use_siren = config['use_siren'] if 'use_siren' in config else False

"""=============== START ==============="""

# Load Dataset
log_path = os.path.join(output_path, experiment_name)
dataset_type = ['train', 'val', 'test']
images, poses, width, height, focal, train_idx = load_blender_data(data_path, data_resize, data_skip, data_view_dir_range, None, data_train_idx)
if data_show_distribution:
    show_data_distribution(poses)
for t in dataset_type:
    if t == 'val':
        images['val']['in'][..., :3] = images['val']['in'][..., :3] * images['val']['in'][..., -1:] + (1. - images['val']['in'][..., -1:])
        images['val']['ex'][..., :3] = images['val']['ex'][..., :3] * images['val']['ex'][..., -1:] + (1. - images['val']['ex'][..., -1:])
    else:
        images[t][..., :3] = images[t][..., :3] * images[t][..., -1:] + (1. - images[t][..., -1:])
print('Data Loaded:\n'
      f'train_set={images[dataset_type[0]].shape}\n'
      f'val_set_in={images[dataset_type[1]]["in"].shape}\n'
      f'val_set_ex={images[dataset_type[1]]["ex"].shape}\n'
      f'test_set={images[dataset_type[2]].shape}\n'
      )

# Model
if use_siren:
    coarse_model = SirenNeRF()
    fine_model = SirenNeRF() if use_fine_model else coarse_model
else:
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
loss_fn_alex = lpips.LPIPS(net='alex')

data = {
    'dist': {'train':[], 'in': [], 'ex': []},
    'psnr': {'train':[], 'in': [], 'ex': []},
    'ssim': {'train':[], 'in': [], 'ex': []},
    'lpips': {'train':[], 'in': [], 'ex': []},
}

data_images = []
data_targets = []

for p in ['train', 'in', 'ex']:
    print(f'Testing {p}terpolate...')
    for pose, target in tqdm(zip(poses['train'], images['train']) if p == 'train' else zip(poses['val'][p], images['val'][p])):
        with torch.no_grad():
            image, _, _ = render_image(width, height, focal, pose, render_near, render_far,
                                 coarse_model, fine_model,
                                 render_coarse_sample_num, render_fine_sample_num
                                 )
            data_images.append(image)
            data_targets.append(target[..., :3])
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
            target = torch.tensor(target[..., :3]).permute(2, 0, 1).unsqueeze(0)
            _, theta, phi = transform_matrix_to_camera_pos(pose)
            dist = max(abs(theta), abs(phi)) if abs(theta) < 90 else 180 - max(180 - abs(theta), abs(phi))
            mse = torch.mean((image - target)**2)
            psnr = -10 * torch.log10(mse).item()
            ssim = pytorch_ssim.ssim(image, target).item()
            lpips = loss_fn_alex(image, target).item()
            data['dist'][p].append(dist)
            data['psnr'][p].append(psnr)
            data['ssim'][p].append(ssim)
            data['lpips'][p].append(lpips)
            tqdm.write(f"[Test] DIST: {dist} PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips}")

test_file_path = os.path.join(log_path, 'test.json')
with open(test_file_path, 'w') as test_file:
    json.dump(data, test_file)
print('Test data write to:', test_file_path)

test_image_path = os.path.join(log_path, 'test.jpg')
data_image = np.concatenate([np.concatenate(data_images, 0), np.concatenate(data_targets, 0)], 1)
imageio.imwrite(test_image_path, to8b(data_image))
print('Test image write to:', test_image_path)
