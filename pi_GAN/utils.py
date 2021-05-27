import torch
import numpy as np
import imageio
import time
import logging
import plyfile
import skimage.measure

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# TRAIN

def summary_module(module):
    # print(list(module.named_modules()))
    total_params = sum(p.numel() for p in module.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in module.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def loss_f(u):
    return -torch.nn.functional.softplus(-u)


def loss_r1(y, x):
    batch_size = x.shape[0]
    gradients = torch.autograd.grad(y, [x], torch.ones_like(y), create_graph=True)[0]
    gradients = gradients.reshape(batch_size, -1)
    res = torch.mean(gradients.norm(dim=-1) ** 2)
    return res


# EXTRACT MESH

def create_mesh(
        generator, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    z = torch.randn(1, generator.input_dim, device='cuda')
    film_params = generator.get_mapping(z)
    generator.set_film_params(film_params[0])
    decoder = generator.film_siren_nerf

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-0.1, -0.1, -0.1]
    voxel_size = 0.2 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.floor_divide(overall_index.long(), N) % N
    samples[:, 0] = torch.floor_divide(torch.floor_divide(overall_index.long(), N), N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        print(head)
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()
        sample_subset = torch.cat([sample_subset, torch.zeros_like(sample_subset)], dim=-1)

        samples[head: min(head + max_batch, num_samples), 3] = (
            -decoder(sample_subset)[:, 3]
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
        pytorch_3d_sdf_tensor,
        voxel_grid_origin,
        voxel_size,
        ply_filename_out,
        offset=None,
        scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        print((np.min(numpy_3d_sdf_tensor[..., 3]) + np.max(numpy_3d_sdf_tensor[..., 3])) / 2)
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=-20, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )


# DEMO

@torch.no_grad()
def save_demo(generator, file_name, rows=4, columns=4, chunk_size=16):
    num = rows * columns
    z = torch.randn(num, generator.input_dim, device='cuda')
    film_params = []
    for i in range(0, num, chunk_size):
        batch_z = z[i:i + chunk_size]
        w = generator.get_mapping(batch_z)
        film_params.append(w)
    film_params = torch.cat(film_params, dim=0)
    gen_image = []
    for i in range(film_params.shape[0]):
        generator.set_film_params(film_params[i])
        gen_image.append(generator.render().cpu().numpy())
    gen_image_row = []
    for i in range(0, num, columns):
        gen_image_row.append(np.concatenate(gen_image[i:i + columns], axis=1))
    demo_image = np.concatenate(gen_image_row, axis=0)
    rgb = to8b(demo_image)
    imageio.imsave(file_name, rgb)


@torch.no_grad()
def demo_multiview(generator, file_name, poses, rows=4, film_params=None, chunk_size=16):
    if film_params is None:
        z = torch.randn(rows, generator.input_dim, device='cuda')
        film_params = []
        for i in range(0, rows, chunk_size):
            batch_z = z[i:i + chunk_size]
            w = generator.get_mapping(batch_z)
            film_params.append(w)
        film_params = torch.cat(film_params, dim=0)
    gen_image_row = []
    for i in range(film_params.shape[0]):
        gen_image = []
        generator.set_film_params(film_params[i])
        for pose in poses:
            if len(pose) >= 3:
                generator.renderer.set_params(fov=pose[2])
            gen_image.append(generator.render(*pose[:2]).cpu().numpy())
        gen_image_row.append(np.concatenate(gen_image, axis=1))
    demo_image = np.concatenate(gen_image_row, axis=0)
    rgb = to8b(demo_image)
    imageio.imsave(file_name, rgb)


@torch.no_grad()
def demo_video(generator, file_name, poses, film_params=None, chunk_size=16):
    if film_params is None:
        z = torch.randn(1, generator.input_dim, device='cuda')
        film_params = generator.get_mapping(z)
    gen_image = []
    generator.set_film_params(film_params[0])
    for pose in poses:
        if len(pose) >= 3:
            generator.renderer.set_params(fov=pose[2])
        gen_image.append(generator.render(*pose[:2]).cpu().numpy())
    video = np.stack(gen_image)
    imageio.mimwrite(file_name, to8b(video), duration=0.1)


@torch.no_grad()
def demo_interpolate(generator, file_name, cols, pose=[0, 0], chunk_size=16):
    z_ = torch.randn(2, generator.input_dim, device='cuda')
    z = []
    film_params = []
    k = np.linspace(0, 1, cols)
    for k_ in k:
        z.append(z_[1] * k_ + z_[0] * (1 - k_))
    z = torch.stack(z)
    for i in range(0, cols, chunk_size):
        batch_z = z[i:i + chunk_size]
        w = generator.get_mapping(batch_z)
        film_params.append(w)
    film_params = torch.cat(film_params, dim=0)
    gen_image_z = []
    gen_image_w = []
    for i in range(cols):
        generator.set_film_params(film_params[i])
        gen_image_z.append(generator.render(*pose[:2]).cpu().numpy())
    gen_image_z = np.concatenate(gen_image_z, axis=1)
    for i in range(cols):
        generator.set_film_params(film_params[0] * (1 - k[i]) + film_params[-1] * k[i])
        gen_image_w.append(generator.render(*pose[:2]).cpu().numpy())
    gen_image_w = np.concatenate(gen_image_w, axis=1)
    demo_image = np.concatenate([gen_image_z, gen_image_w], axis=0)
    rgb = to8b(demo_image)
    imageio.imsave(file_name, rgb)


@torch.no_grad()
def demo_style_mix(generator, file_name, rows, pose=[0, 0], chunk_size=16):
    num = 2 * rows
    z = torch.randn(num, generator.input_dim, device='cuda')
    film_params = []
    for i in range(0, num, chunk_size):
        batch_z = z[i:i + chunk_size]
        w = generator.get_mapping(batch_z)
        film_params.append(w)
    film_params = torch.cat(film_params, dim=0)
    gen_image_row = []
    for i in range(rows):
        gen_image = []
        for k in range(9, -1, -1):
            film_params_ = torch.cat([film_params[2 * i][:k], film_params[2 * i + 1][k:]], dim=0)
            generator.set_film_params(film_params_)
            gen_image.append(generator.render(*pose[:2]).cpu().numpy())
        gen_image_row.append(np.concatenate(gen_image, axis=1))
    demo_image = np.concatenate(gen_image_row, axis=0)
    rgb = to8b(demo_image)
    imageio.imsave(file_name, rgb)
