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

z_dim = config['z_dim'] if 'z_dim' in config else 1024
iterations = config['iterations'] if 'iterations' in config else [50000]
fade_in_itrs = config['fade_in_itrs'] if 'fade_in_itrs' in config else [0]
batch_size = config['batch_size'] if 'batch_size' in config else [64]
resolution = config['resolution'] if 'resolution' in config else [32]
generator_lr = config['generator_lr'] if 'generator_lr' in config else 5e-5
discriminator_lr = config['discriminator_lr'] if 'discriminator_lr' in config else 4e-4
generator_lr_end = config['generator_lr_end'] if 'generator_lr_end' in config else 1e-5
discriminator_lr_end = config['discriminator_lr_end'] if 'discriminator_lr_end' in config else 1e-4
lr_decay = config['lr_decay'] if 'lr_decay' in config else 500

i_print = config['i_print'] if 'i_print' in config else 100
i_save = config['i_save'] if 'i_save' in config else 10000
i_image = config['i_image'] if 'i_image' in config else 1000

iterations = [0] + iterations


"""=============== START ==============="""
# Model
generator = Generator(z_dim, resolution[0], render_near, render_far, 12, render_coarse_sample_num, render_fine_sample_num, 0.3, 0.15)
generator = torch.nn.DataParallel(generator)
discriminator = Discriminator(resolution[0])
discriminator = torch.nn.DataParallel(discriminator)
g_optimizer = torch.optim.Adam(params=generator.parameters(), lr=generator_lr, betas=(0, 0.9))
d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=discriminator_lr, betas=(0, 0.9))
summary_module(generator)
summary_module(discriminator)
# renderer.show_distribution()

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
    g_optimizer.load_state_dict(check_point['g_optimizer'])
    d_optimizer.load_state_dict(check_point['d_optimizer'])
    generator.module.load_state_dict(check_point['generator'])
    discriminator.module.load_state_dict(check_point['discriminator'])
else:
    global_step = 0
    loss_log = {'g_loss': [], 'd_loss': []}
global_step += 1

stage = 0
for i in range(len(iterations)):
    if global_step > iterations[i]:
        stage = i
    else:
        break
dataset = DataLoader(data_path, batch_size[stage], resize=resolution[stage]/64, preload=False)
generator.module.set_resolution(resolution[stage])
discriminator.module.set_resolution(resolution[stage])
print(f'Starting at stage {stage}, batch_size:{batch_size[stage]}, resolution:{resolution[stage]}')

# Training
start = global_step
for global_step in trange(start, iterations[-1] + 1):
    epoch_idx, batch_idx, real_image = dataset.get()

    # fade in
    fade_in_alpha = -1
    if global_step < iterations[stage] + fade_in_itrs[stage]:
        fade_in_alpha = (global_step - iterations[stage]) / fade_in_itrs[stage]

    ## train D
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    # real
    real_image = real_image.permute(0, 3, 1, 2).contiguous()
    real_image.requires_grad = True
    real_label = discriminator(real_image, alpha=fade_in_alpha)

    # generate
    z = torch.randn(batch_size[stage], z_dim, device='cuda')
    gen_image = generator(z)
    gen_label = discriminator(gen_image, alpha=fade_in_alpha)

    # optimize
    d_optimizer.zero_grad()
    lambda_ = 10
    d_loss = -torch.mean(loss_f(gen_label)) - torch.mean(loss_f(-real_label)) + lambda_ * loss_r1(real_label, real_image)
    d_loss.backward()
    d_optimizer.step()
    loss_log['d_loss'].append(d_loss.item())

    ## train G
    requires_grad(generator, True)
    requires_grad(discriminator, False)

    # generate
    z = torch.randn(batch_size[stage], z_dim, device='cuda')
    gen_image = generator(z)
    gen_label = discriminator(gen_image, alpha=fade_in_alpha)

    # optimize
    g_optimizer.zero_grad()
    g_loss = torch.mean(loss_f(gen_label))
    g_loss.backward()
    g_optimizer.step()
    loss_log['g_loss'].append(g_loss.item())

    # NOTE: IMPORTANT!
    ###   update learning rate   ###
    decay_rate = 0.1
    decay_steps = lr_decay * 1000
    new_g_lr = generator_lr_end + (generator_lr - generator_lr_end) * (decay_rate ** (global_step / decay_steps))
    new_d_lr = discriminator_lr_end + (discriminator_lr - discriminator_lr_end) * (decay_rate ** (global_step / decay_steps))
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = new_g_lr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = new_d_lr

    # Stage
    if global_step == iterations[stage + 1]:
        stage += 1
        if stage + 1 < len(iterations):
            dataset = DataLoader(data_path, batch_size[stage], resize=resolution[stage] / 64, preload=False)
            generator.module.set_resolution(resolution[stage])
            discriminator.module.set_resolution(resolution[stage])
            tqdm.write(f'[Train] Entering stage {stage}, batch_size:{batch_size[stage]}, resolution:{resolution[stage]}')

    # Logging
    if global_step % i_print == 0:
        tqdm.write(f"[Train] Iter: {global_step}({epoch_idx}-{batch_idx}) d_loss: {d_loss.item()} g_loss: {g_loss.item()}")

    if global_step % i_save == 0:
        path = os.path.join(output_path, experiment_name, '{:06d}.tar'.format(global_step))
        torch.save({
            'global_step': global_step,
            'loss_log': loss_log,
            'generator': generator.module.state_dict(),
            'discriminator': discriminator.module.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
        }, path)
        tqdm.write(f'Saved checkpoints at {path}')

    if global_step % i_image == 0:
        # Turn on testing mode
        filename = os.path.join(log_path, '{:06d}.png'.format(global_step))
        save_demo(generator.module, filename)
