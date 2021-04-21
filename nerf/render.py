import numpy as np

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


