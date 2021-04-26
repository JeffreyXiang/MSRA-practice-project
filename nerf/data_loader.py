import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import json

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


def load_blender_data(file_path, resize=1, test_skip=1, view_dir_range=None, target_num=None, train_idx=None):
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
    train_idx_res = []
    for t in file_type:
        meta = metas[t]
        type_images = []
        type_poses = []
        type_images_ = []
        type_poses_ = []
        skip = 1 if t != 'test' or test_skip == 0 else test_skip

        for frame in meta['frames'][::skip]:
            _, theta, phi = transform_matrix_to_camera_pos(blender_coord @ np.array(frame['transform_matrix']))
            flag = False
            if t == 'test':
                flag = True
            elif t == 'val' or train_idx is None:
                if view_dir_range is None:
                    flag = True
                else:
                    for r in view_dir_range:
                        if r[0] < theta < r[1] and r[2] < phi < r[3]:
                            flag = True
                            break
            else:
                file_idx = int(frame['file_path'].split('_')[1])
                if file_idx in train_idx:
                    flag = True
            if flag:
                if t == 'train':
                    file_idx = int(frame['file_path'].split('_')[1])
                    train_idx_res.append(file_idx)
                file_name = os.path.join(file_path, frame['file_path'] + '.png')
                image = Image.open(file_name)
                if resize != 1:
                    image = image.resize((int(resize * image.width), int(resize * image.height)), Image.ANTIALIAS)
                type_images.append(np.array(image, dtype=np.float32))
                type_poses.append(blender_coord @ np.array(frame['transform_matrix'], dtype=np.float32))
            elif t == 'val':
                type_images_.append(np.array(image, dtype=np.float32))
                type_poses_.append(blender_coord @ np.array(frame['transform_matrix'], dtype=np.float32))

        type_images = (np.array(type_images) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        type_poses = np.array(type_poses).astype(np.float32)
        if t == 'train' and target_num is not None:
            choice_idx = np.random.choice(list(range(type_images.shape[0])), size=target_num, replace=False)
            type_images = type_images[choice_idx]
            type_poses = type_poses[choice_idx]
            for i in reversed(range(len(train_idx_res))):
                if i not in choice_idx:
                    del train_idx_res[i]
        if t == 'val':
            type_images_ = (np.array(type_images_) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            type_poses_ = np.array(type_poses_).astype(np.float32)
            images['val'] = {'in': type_images, 'ex': type_images_}
            poses['val'] = {'in': type_poses, 'ex': type_poses_}
        else:
            images[t] = type_images
            poses[t] = type_poses

    height, width = images['train'][0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    return images, poses, width, height, focal, train_idx_res


def show_data_distribution(poses, show_test=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = poses['train'][:, 0, 3]
    ys = poses['train'][:, 1, 3]
    zs = poses['train'][:, 2, 3]
    ax.scatter(xs, ys, zs, c='r', marker='o')

    if poses['val']['in'].shape[0] > 0:
        xs = poses['val']['in'][:, 0, 3]
        ys = poses['val']['in'][:, 1, 3]
        zs = poses['val']['in'][:, 2, 3]
        ax.scatter(xs, ys, zs, c='g', marker='s')

    if poses['val']['ex'].shape[0] > 0:
        xs = poses['val']['ex'][:, 0, 3]
        ys = poses['val']['ex'][:, 1, 3]
        zs = poses['val']['ex'][:, 2, 3]
        ax.scatter(xs, ys, zs, c='b', marker='s')

    if show_test:
        xs = poses['test'][:, 0, 3]
        ys = poses['test'][:, 1, 3]
        zs = poses['test'][:, 2, 3]
        ax.scatter(xs, ys, zs, c='y', marker='^')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=90, azim=-90)

    plt.show()
