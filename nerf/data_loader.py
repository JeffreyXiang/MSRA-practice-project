import os
import numpy as np
import PIL.Image as Image
import json

# z-axis translation matrix
trans_t = lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
], dtype=float)

# pitch rotation matrix (+:down, -:up)
rot_phi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=float)

# yaw rotation matrix (+:right, -:left)
rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]
], dtype=float)

blender_coord = np.array([
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=float)


def camera_pos_to_transform_matrix(radius, theta, phi):
    """
    Get transform matrix with camera position

    :param theta:
    :param phi:
    :param radius:
    :return: camera to world transform matrix
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    return c2w


def transform_matrix_to_camera_pos(c2w):
    """
    Get camera position with transform matrix

    :param c2w: camera to world transform matrix
    :return: camera position on spherical coord
    """

    pos = c2w @ np.array([[0.], [0.], [0.], [1.]]).squeeze()
    radius = float(np.linalg.norm(pos[:-1]))
    theta = float(np.arctan2(-pos[0], pos[2])) / np.pi * 180
    phi = float(np.arctan(-pos[1] / np.linalg.norm(pos[::2]))) / np.pi * 180
    return radius, theta, phi


def load_blender_data(file_path, resize=1, test_skip=1, view_dir_range=None):
    """
    Get the Blender data from given directory

    :param file_path: directory path
    :param resize: resize ratio
    :param test_skip: skip step when getting test and validation data
    :param view_dir_range:
    :return:
    """
    file_type = ['train', 'val', 'test']
    metas = {}
    for t in file_type:
        with open(os.path.join(file_path, 'transforms_{}.json'.format(t)), 'r') as fp:
            metas[t] = json.load(fp)

    images = {}
    poses = {}
    for t in file_type:
        meta = metas[t]
        type_images = []
        type_poses = []
        skip = 1 if t == 'train' or test_skip == 0 else test_skip

        for frame in meta['frames'][::skip]:
            _, theta, phi = transform_matrix_to_camera_pos(blender_coord @ np.array(frame['transform_matrix']))
            flag = False
            if view_dir_range is None:
                flag = True
            else:
                for r in view_dir_range:
                    if r[0] < theta < r[1] and r[2] < phi < r[3]:
                        flag = True
                        break
            if flag:
                print(frame['file_path'])
                file_name = os.path.join(file_path, frame['file_path'] + '.png')
                image = Image.open(file_name)
                if resize != 1:
                    image = image.resize((int(resize * image.width), int(resize * image.height)), Image.ANTIALIAS)
                type_images.append(np.array(image))
                type_poses.append(blender_coord @ np.array(frame['transform_matrix']))

        type_images = (np.array(type_images) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        type_poses = np.array(type_poses).astype(np.float32)
        images[t] = type_images
        poses[t] = type_poses

    height, width = images['train'][0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    return images, poses, width, height, focal
