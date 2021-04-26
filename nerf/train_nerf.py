import torch
import os
import sys
import json
from tqdm import tqdm, trange
import imageio
from data_loader import *
from render import *
from nerf import NeRF

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = sys.argv[1]
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']
data_path = config['data_path']
data_resize = config['data_resize'] if 'data_resize' in config else 0.5
data_skip = config['data_skip'] if 'data_skip' in config else 8
data_train_idx = config['data_train_idx'] if 'data_train_idx' in config else None
data_view_dir_range = config['data_view_dir_range'] if 'data_view_dir_range' in config else None
data_view_dir_noise = config['data_view_dir_noise'] if 'data_view_dir_noise' in config else None
data_target_num = config['data_target_num'] if 'data_target_num' in config else None
data_show_distribution = config['data_show_distribution'] if 'data_show_distribution' in config else False

render_near = config['render_near'] if 'render_near' in config else 2.0
render_far = config['render_far'] if 'render_far' in config else 6.0
render_coarse_sample_num = config['render_coarse_sample_num'] if 'render_coarse_sample_num' in config else 64
render_fine_sample_num = config['render_fine_sample_num'] if 'render_fine_sample_num' in config else 128

iterations = config['iterations'] if 'iterations' in config else 200000
batch_size = config['batch_size'] if 'batch_size' in config else 1024
learning_rate = config['learning_rate'] if 'learning_rate' in config else 5e-4
learning_rate_decay = config['learning_rate_decay'] if 'learning_rate_decay' in config else 500
start_up_itrs = config['start_up_itrs'] if 'start_up_itrs' in config else 500
use_fine_model = config['use_fine_model'] if 'use_fine_model' in config else True
use_alpha = config['use_alpha'] if 'use_alpha' in config else False

i_print = config['i_print'] if 'i_print' in config else 100
i_save = config['i_save'] if 'i_save' in config else 10000
i_image = config['i_image'] if 'i_image' in config else 1000

"""=============== START ==============="""

# Load Dataset
log_path = os.path.join(output_path, experiment_name)
os.makedirs(log_path, exist_ok=True)

dataset_type = ['train', 'val', 'test']
images, poses, width, height, focal, train_idx = load_blender_data(data_path, data_resize, data_skip, data_view_dir_range, data_target_num, data_train_idx)
config['data_train_idx'] = train_idx
config_file_path = os.path.join(log_path, 'config.json')
with open(config_file_path, 'w') as config_file:
    json.dump(config, config_file)
print('Config file write to:', config_file_path)
if data_show_distribution:
    show_data_distribution(poses)
for t in dataset_type:
    if t == 'val':
        images['val']['in'][..., :3] = images['val']['in'][..., :3] * images['val']['in'][..., -1:] + (1. - images['val']['in'][..., -1:])
        images['val']['ex'][..., :3] = images['val']['ex'][..., :3] * images['val']['ex'][..., -1:] + (1. - images['val']['ex'][..., -1:])
    else:
        images[t][..., :3] = images[t][..., :3] * images[t][..., -1:] + (1. - images[t][..., -1:])
if data_view_dir_noise is not None:
    poses['train'] += np.random.normal(size=poses['train'].shape) * np.sqrt(data_view_dir_noise)
print('Data Loaded:\n'
      f'train_set={images[dataset_type[0]].shape}\n'
      f'val_set_in={images[dataset_type[1]]["in"].shape}\n'
      f'val_set_ex={images[dataset_type[1]]["ex"].shape}\n'
      f'test_set={images[dataset_type[2]].shape}\n'
      )

# Batching
rays = np.stack([get_rays(width, height, focal, p) for p in poses[dataset_type[0]][:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd, 3]
rays = np.reshape(rays, [-1, 6])  # [N*H*W, 6]
rgba = np.reshape(images[dataset_type[0]], [-1, 4]) # [N*H*W, 4]
rays_rgba = np.concatenate([rays, rgba], 1) # [N*H*W, 10]
np.random.shuffle(rays_rgba)
rays_rgba = torch.tensor(rays_rgba, dtype=torch.float, device='cuda')
batch_num = int(np.ceil(rays_rgba.shape[0] / batch_size))
print(f'Batching Finished: size={rays_rgba.shape}, batch_size={batch_size}, batch_num={batch_num}')

# Model
coarse_model = NeRF()
fine_model = NeRF() if use_fine_model else coarse_model
trainable_variables = list(coarse_model.parameters())
if use_fine_model:
    trainable_variables += list(fine_model.parameters())
optimizer = torch.optim.Adam(params=trainable_variables, lr=learning_rate, betas=(0.9, 0.999))

# Load log directory
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)

    global_step = check_point['global_step']
    optimizer.load_state_dict(check_point['optimizer'])
    coarse_model.load_state_dict(check_point['coarse_model'])
    if use_fine_model:
        fine_model.load_state_dict(check_point['fine_model'])
else:
    global_step = 0

# Training
s_width = int(width / 2)
s_height = int(height / 2)
s_left = int(width / 4)
s_top = int(height / 4)
batch_idx = 0
global_step += 1
start = global_step
for global_step in trange(start, iterations + 1):
    if global_step <= start_up_itrs:  # Start up
        if global_step == 1:
            tqdm.write(f"[Train] Start-up phase with {start_up_itrs} iterations.")
        s_image_idx = np.random.choice(range(images['train'].shape[0]))
        s_image = images['train'][s_image_idx]
        s_pose = poses['train'][s_image_idx]
        s_rays = np.concatenate([get_rays(s_width, s_height, focal, s_pose[:3, :4])])  # [ro+rd, H, W, 3]
        s_rays = np.transpose(s_rays, [1, 2, 0, 3])  # [H, W, ro+rd, 3]
        s_rays = np.reshape(s_rays, [-1, 6])  # [H*W, 6]
        s_rgba = np.reshape(s_image[s_top:s_top + s_height, s_left:s_left + s_width], [-1, 4])  # [H*W, 4]
        s_rays_rgba = np.concatenate([s_rays, s_rgba], 1)  # [H*W, 10]
        s_sample_idx = np.random.choice(range(s_rays_rgba.shape[0]), size=batch_size, replace=False)
        batch = torch.tensor(s_rays_rgba[s_sample_idx], dtype=torch.float, device='cuda')
    else:
        batch = rays_rgba[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_idx += 1
        if batch_idx == batch_num:
            # Shuffle data at the beginning of a epoch
            shuffle_idx = torch.randperm(rays_rgba.shape[0])
            rays_rgb = rays_rgba[shuffle_idx]
            batch_idx = 0

    batch_rays = torch.reshape(batch[:, :6], [-1, 2, 3])
    batch_rgb = batch[:, -4:-1]
    batch_alpha = batch[:, -1]
    # Render
    rgb_map_coarse, _, acc_map_coarse, rgb_map_fine, _, acc_map_fine =\
        render_rays(batch_rays, render_near, render_far,
                    coarse_model, fine_model,
                    render_coarse_sample_num, render_fine_sample_num)

    # Optimize
    optimizer.zero_grad()
    loss_coarse = torch.mean((rgb_map_coarse - batch_rgb) ** 2)
    loss_fine = torch.mean((rgb_map_fine - batch_rgb) ** 2)
    psnr = -10 * torch.log10(loss_fine)
    if use_alpha:
        loss_coarse += 0.1 * torch.mean((acc_map_coarse - batch_alpha) ** 2)
        loss_fine += 0.1 * torch.mean((acc_map_fine - batch_alpha) ** 2)
    loss = loss_fine
    if use_fine_model:
        loss += loss_coarse
    loss.backward()
    optimizer.step()

    # NOTE: IMPORTANT!
    ###   update learning rate   ###
    decay_rate = 0.1
    decay_steps = learning_rate_decay * 1000
    new_learning_rate = learning_rate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate

    if global_step % i_print == 0:
        tqdm.write(f"[Train] Iter: {global_step} Loss: {loss.item()} PSNR: {psnr.item()}")

    if global_step % i_save == 0:
        path = os.path.join(output_path, experiment_name, '{:06d}.tar'.format(global_step))
        torch.save({
            'global_step': global_step,
            'coarse_model': coarse_model.state_dict(),
            'fine_model': fine_model.state_dict() if use_fine_model else None,
            'optimizer': optimizer.state_dict(),
        }, path)
        tqdm.write(f'Saved checkpoints at {path}')

    if global_step % i_image == 0:
        # Turn on testing mode
        with torch.no_grad():
            image, _, _ = render_image(width, height, focal, camera_pos_to_transform_matrix(4, 0, 0),
                                       render_near, render_far,
                                       coarse_model, fine_model,
                                       render_coarse_sample_num, render_fine_sample_num
                                       )
        rgb8 = to8b(image)
        filename = os.path.join(log_path, '{:06d}.png'.format(global_step))
        imageio.imwrite(filename, rgb8)
