import torch
from tqdm import tqdm, trange
import imageio
from data_loader import *
from render import *
from nerf import NeRF

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
output_path = './logs/'
experiment_name = 'lego_1'
data_path = '../../nerf-pytorch/data/nerf_synthetic/lego'
data_resize = 0.5
data_skip = 8

render_near = 2.0
render_far = 6.0
render_coarse_sample_num = 64
render_fine_sample_num = 128

iterations = 200000
batch_size = 1024
learning_rate = 5e-4
learning_rate_decay = 500

i_print = 100
i_save = 10000
i_video = 1000

"""=============== START ==============="""

# Load Dataset
dataset_type = ['train', 'val', 'test']
images, poses, width, height, focal = load_blender_data(data_path, data_resize, data_skip)
for t in dataset_type:
    images[t] = images[t][..., :3] * images[t][..., -1:] + (1. - images[t][..., -1:])
print('Data Loaded:\n'
      f'train_set={images[dataset_type[0]].shape}\n'
      f'val_set={images[dataset_type[1]].shape}\n'
      f'test_set={images[dataset_type[2]].shape}\n'
      )

# Batching
rays = np.stack([get_rays(width, height, focal, p) for p in poses[dataset_type[0]][:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
rays_rgb = np.concatenate([rays, images[dataset_type[0]][:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [N*H*W, ro+rd+rgb, 3]
np.random.shuffle(rays_rgb)
rays_rgb = torch.tensor(rays_rgb, dtype=torch.float, device='cuda')
batch_num = rays_rgb.shape[0] // batch_size + 1
print(f'Batching Finished: size={rays_rgb.shape}, batch_size={batch_size}, batch_num={batch_num}')

# Model
coarse_model = NeRF()
fine_model = NeRF()
trainable_variables = list(coarse_model.parameters()) + list(fine_model.parameters())
optimizer = torch.optim.Adam(params=trainable_variables, lr=learning_rate, betas=(0.9, 0.999))

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
    coarse_model.load_state_dict(check_point['coarse_model'])
    fine_model.load_state_dict(check_point['fine_model'])

# Training
global_step = 0
batch_idx = 0
global_step += 1
start = global_step
for global_step in trange(start, iterations + 1):
    batch = rays_rgb[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_rays, batch_rgb = batch[:, :2], batch[:, 2].squeeze()
    batch_idx += 1
    if batch_idx == batch_num:
        # Shuffle data at the beginning of a epoch
        shuffle_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[shuffle_idx]
        batch_idx = 0

    # Render
    rgb_map_coarse, _, _, rgb_map_fine, _, _ =\
        render_rays(batch_rays, render_near, render_far,
                    coarse_model, fine_model,
                    render_coarse_sample_num, render_fine_sample_num)

    # Optimize
    optimizer.zero_grad()
    loss_coarse = torch.mean((rgb_map_coarse - batch_rgb) ** 2)
    loss_fine = torch.mean((rgb_map_fine - batch_rgb) ** 2)
    loss = loss_coarse + loss_fine
    psnr = -10 * torch.log10(loss_fine)
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
            'fine_model': fine_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, path)
        tqdm.write(f'Saved checkpoints at {path}')

    if global_step % i_video == 0:
        # Turn on testing mode
        with torch.no_grad():
            image = render_image(height, width, focal, camera_pos_to_transform_matrix(4, 0, 0),
                                 render_near, render_far,
                                 coarse_model, fine_model,
                                 render_coarse_sample_num, render_fine_sample_num
                                 )
        rgb8 = to8b(image)
        filename = os.path.join(log_path, '{:06d}.png'.format(global_step))
        imageio.imwrite(filename, rgb8)
