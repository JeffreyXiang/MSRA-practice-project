import torch
import numpy as np

to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)

def render_image(model, width, height, chunk=8192):
    with torch.no_grad():
        pos = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        pos = np.concatenate([pos[0].reshape((-1, 1)), pos[1].reshape((-1, 1))], axis=1)
        pos = torch.tensor(pos, dtype=torch.float, device='cuda')
        rgb = np.zeros((width * height, 1), dtype=float)
        for i in range(0, width * height, chunk):
            chunk_pos = pos[i:i + chunk]
            rgb[i:i + chunk] = model(chunk_pos).cpu().numpy()
        rgb = rgb.reshape((height, width, 1))
    return rgb
