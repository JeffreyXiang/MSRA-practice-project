import torch
from tqdm import tqdm
import numpy as np

# z-axis translation matrix
trans_t = lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
], dtype=np.float32)

# pitch rotation matrix (+:down, -:up)
rot_phi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# yaw rotation matrix (+:right, -:left)
rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]
], dtype=np.float32)

blender_coord = np.array([
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)


def camera_pos_to_transform_matrix(radius, theta, phi):
    """
    Get transform matrix with camera position

    :param theta:
    :param phi:
    :param radius:
    :return: camera to world transform matrix
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    return c2w


def get_rays(width, height, focal, c2w):
    """
    Get rays for every pixel in the image

    :param width: image width
    :param height: image height
    :param focal: image focal length
    :param c2w: camera to world transform matrix
    :return: numpy array of ray origin (camera position) and ray direction
    """
    i, j = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - width * 0.5) / focal, -(j - height * 0.5) / focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    u = torch.linspace(0., 1., steps=N_samples)
    u = u.expand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def run_network(ray_samples, view_dirs, network, chunk=1024*64):
    """
    Prepares inputs and applies network

    :param ray_samples: sample points along each ray [N_rays, N_samples, 3]
    :param view_dirs: view direction of each ray [N_rays, 3]
    :param network:
    :param chunk: max points input to the model at a time
    :return:
    """
    ray_samples_flat = torch.reshape(ray_samples, [-1, 3])
    view_dirs = view_dirs[:,None].expand(ray_samples.shape)
    view_dirs_flat = torch.reshape(view_dirs, [-1, 3])
    inputs = torch.cat([ray_samples_flat, view_dirs_flat], -1)
    outputs = torch.cat([network(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)])
    outputs = torch.reshape(outputs, list(ray_samples.shape[:-1]) + [4])
    return outputs


def raw_to_outputs(raw, z_vals, rays_d):
    """
        Transforms model's predictions to semantically meaningful values.

        :param raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        :param z_vals: [num_rays, num_samples along ray]. Integration time.
        :param rays_d: [num_rays, 3]. Direction of each ray.
        :return:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            depth_map: [num_rays]. Estimated distance to object.
            acc_map: [num_rays]. Sum of weights along each ray.
    """

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([[1e10]]).expand([dists.shape[0], 1])], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)

    rgb = raw[...,:3]  # [N_rays, N_samples, 3]
    alpha = 1.0 - torch.exp(-raw[...,3] * dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def render_rays(rays, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num):
    """
    Render the color of the ray (Volume Render)

    :param rays:
    :param near: near plane
    :param far: far plane
    :param coarse_model:
    :param fine_model:
    :param coarse_sample_num:
    :param fine_sample_num:
    :return:
    """

    rays_o = rays[:, 0].squeeze()
    rays_d = rays[:, 1].squeeze()
    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    z_vals = torch.linspace(near, far, steps=coarse_sample_num).unsqueeze(0).expand([rays_o.shape[0], coarse_sample_num])

    ### Render coarse model
    # Get intervals between samples
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # Stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape)
    z_vals = lower + (upper - lower) * t_rand
    # Final coarse sample points
    coarse_samples = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = run_network(coarse_samples, view_dirs, coarse_model)
    rgb_map_coarse, depth_map_coarse, acc_map_coarse, weights = raw_to_outputs(raw, z_vals, rays_d)

    ### Render fine model
    # Hierarchical volume sampling
    z_samples = sample_pdf(mids, weights[..., 1:-1], fine_sample_num)
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
    fine_samples = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
    raw = run_network(fine_samples, view_dirs, fine_model)
    rgb_map_fine, depth_map_fine, acc_map_fine, _ = raw_to_outputs(raw, z_vals, rays_d)

    return rgb_map_coarse, depth_map_coarse, acc_map_coarse, rgb_map_fine, depth_map_fine, acc_map_fine


def render_image(width, height, focal, pose, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num, chunk=1024*16):
    rays = get_rays(width, height, focal, pose)
    rays = np.stack(rays, 0)
    rays = np.transpose(rays, [1, 2, 0, 3])
    rays = np.reshape(rays, [-1, 2, 3])
    rgb_image = []
    for i in range(0, rays.shape[0], chunk):
        chunk_rays = torch.tensor(rays[i:i+chunk], dtype=torch.float)
        _, _, _, rgb_map, _, _ = render_rays(chunk_rays, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num)
        rgb_image.append(rgb_map)
    rgb_image = torch.cat(rgb_image).reshape([height, width, 3])
    return rgb_image


def render_image_np(width, height, focal, pose, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num, chunk=1024*16):
    rays = get_rays(width, height, focal, pose)
    rays = np.stack(rays, 0)
    rays = np.transpose(rays, [1, 2, 0, 3])
    rays = np.reshape(rays, [-1, 2, 3])
    rgb_image = []
    depth_image = []
    acc_image = []
    for i in range(0, rays.shape[0], chunk):
        chunk_rays = torch.tensor(rays[i:i+chunk], dtype=torch.float)
        _, _, _, rgb_map, depth_map, acc_map = render_rays(chunk_rays, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num)
        rgb_image.append(rgb_map.cpu().numpy())
        depth_image.append(depth_map.cpu().numpy())
        acc_image.append(acc_map.cpu().numpy())
    rgb_image = np.concatenate(rgb_image).reshape([height, width, 3])
    depth_image = np.concatenate(depth_image).reshape([height, width, 1])
    acc_image = np.concatenate(acc_image).reshape([height, width, 1])
    return rgb_image, depth_image, acc_image


def render_video_np(width, height, focal, poses, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num, chunk=1024*16):
    rgb_video = []
    depth_video = []
    acc_video = []
    for _, p in enumerate(tqdm(poses)):
        rgb_image, depth_image, acc_image = render_image(width, height, focal, p, near, far, coarse_model, fine_model, coarse_sample_num, fine_sample_num, chunk)
        rgb_video.append(rgb_image)
        depth_video.append(depth_image)
        acc_video.append(acc_image)
    rgb_video = np.stack(rgb_video)
    depth_video = np.stack(depth_video)
    acc_video = np.stack(acc_video)
    return rgb_video, depth_video, acc_video
