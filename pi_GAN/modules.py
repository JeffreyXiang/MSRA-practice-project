import torch
import numpy as np
from render import *
from matplotlib import pyplot as plt

"""=============== GENERATOR ==============="""

class FilmSiren(torch.nn.Module):
    """Siren layer: Linear layer and sine activation"""

    def __init__(self, input_dim: int, output_dim: int, c=6, w_0=30, is_first_layer=False) -> None:
        super(FilmSiren, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c = c
        self.w_0 = w_0
        self.is_first_layer = is_first_layer
        self.weight = torch.nn.Parameter(torch.zeros(output_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def forward(self, input_tensor, gamma, beta):
        x = torch.nn.functional.linear(input_tensor, self.weight, self.bias)
        x = gamma * x + beta
        return torch.sin(self.w_0 * x)

    def reset_parameters(self) -> None:
        weight_range = 1 / self.input_dim if self.is_first_layer else np.sqrt(self.c / self.input_dim) / self.w_0
        bias_range = np.sqrt(1 / self.input_dim)
        torch.nn.init.uniform_(self.weight, -weight_range, weight_range)
        torch.nn.init.uniform_(self.bias, -bias_range, bias_range)


class MappingNetwork(torch.nn.Module):
    """Mapping Network, convert z to w to condition the NeRF"""

    def __init__(self, input_dim=256, output_dim=256, output_layers=8, hidden_dim=256, hidden_layers=3):
        super(MappingNetwork, self).__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2)
        )
        self.hidden_layers = []
        for i in range(hidden_layers - 1):
            self.hidden_layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(0.2)
            ])
        self.hidden_layers = torch.nn.Sequential(*self.hidden_layers)
        self.output_layers = []
        for i in range(output_layers):
            self.output_layers.append(torch.nn.Linear(hidden_dim, 2 * output_dim))
        self.output_layers.append(torch.nn.Linear(hidden_dim, 2 * output_dim))
        for layer in self.output_layers:
            layer.bias.data[:output_dim] = 1
            layer.bias.data[output_dim:] = 0
        self.output_layers = torch.nn.ModuleList(self.output_layers)

    def forward(self, input_tensor):
        h = self.input_layer(input_tensor)
        h = self.hidden_layers(h)
        output_tensors = []
        for output_layer in self.output_layers:
            output_tensors.append(output_layer(h).unsqueeze(1))
        output_tensors = torch.cat(output_tensors, dim=1)
        return output_tensors

class FilmSirenNeRF(torch.nn.Module):
    """Major part of the generator"""

    def __init__(self, hidden_dim=256, hidden_layers=8, c=6, w_0=30):
        super(FilmSirenNeRF, self).__init__()
        self.film_params = None
        self.input_layer = FilmSiren(3, hidden_dim, c=c, w_0=w_0, is_first_layer=True)
        self.hidden_layers = []
        for i in range(hidden_layers - 1):
            self.hidden_layers.append(FilmSiren(hidden_dim, hidden_dim, c=c, w_0=w_0))
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)
        self.output_layer_sigma = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.ReLU()
        )
        self.hidden_layer_rgb = FilmSiren(hidden_dim + 3, hidden_dim, c=c, w_0=w_0)
        self.output_layer_rgb = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 3),
            torch.nn.Sigmoid()
        )
        self.n_layers = len(self.hidden_layers)

    def set_film_params(self, mapping_tensor):
        self.film_params = []
        for i in range(mapping_tensor.shape[0]):
            self.film_params.append(torch.chunk(mapping_tensor[i], 2))

    def forward(self, input_tensor, film_params=None):
        if film_params is not None:
            self.film_params = film_params
        elif self.film_params is not None:
            film_params = self.film_params
        else:
            raise ValueError
        input_pos, input_dir = torch.split(input_tensor, [3, 3], dim=-1)
        h = self.input_layer(input_pos, *film_params[0])
        for i in range(self.n_layers):
            h = self.hidden_layers[i](h, *film_params[i + 1])
        sigma = self.output_layer_sigma(h)
        h = torch.cat([h, input_dir], -1)
        h = self.hidden_layer_rgb(h, *film_params[self.n_layers + 1])
        rgb = self.output_layer_rgb(h)
        outputs = torch.cat([rgb, sigma], -1)
        return outputs


class Renderer:
    def __init__(self, width, height, near=0.1, far=1.9, fov=12, coarse_samples=64, fine_samples=128,
                 horizontal_std=0.3, vertical_std=0.15):
        self.width = width
        self.height = height
        self.focal = width / 2 / np.tan(fov / 2 * np.pi / 180)
        self.near = near
        self.far = far
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples
        self.horizontal_std = horizontal_std
        self.vertical_std = vertical_std

    def set_params(self, width=None, height=None, near=None, far=None, coarse_samples=None, fine_samples=None,
                   horizontal_std=None, vertical_std=None):
        self.width = width if width is not None else self.width
        self.height = height if height is not None else self.height
        self.near = near if near is not None else self.near
        self.far = far if far is not None else self.far
        self.coarse_samples = coarse_samples if coarse_samples is not None else self.coarse_samples
        self.fine_samples = fine_samples if fine_samples is not None else self.fine_samples
        self.horizontal_std = horizontal_std if horizontal_std is not None else self.horizontal_std
        self.vertical_std = vertical_std if vertical_std is not None else self.vertical_std

    def show_distribution(self):
        theta = np.random.randn(1000) * self.horizontal_std
        phi = np.random.randn(1000) * self.vertical_std
        plt.scatter(theta, phi)
        plt.show()

    def __call__(self, model, theta=None, phi=None):
        if theta is None:
            theta = np.random.randn() * self.horizontal_std
        if phi is None:
            phi = np.random.randn() * self.vertical_std
        pose = camera_pos_to_transform_matrix(1, theta, phi)
        img = render_image(self.width, self.height, self.focal, pose, self.near, self.far, model, model,
                           self.coarse_samples, self.fine_samples)
        return img


class Generator(torch.nn.Module):
    """pi-GAN Generator"""

    def __init__(self, input_dim, output_size, near=0.1, far=1.9, fov=12, coarse_samples=64, fine_samples=128,
                 horizontal_std=0.3, vertical_std=0.15):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.film_siren_nerf = FilmSirenNeRF()
        self.mapping_network = MappingNetwork(input_dim=input_dim)
        self.renderer = Renderer(output_size, output_size, near, far, fov, coarse_samples, fine_samples, horizontal_std, vertical_std)

    def forward(self, input_tensor):
        film_params = self.mapping_network(input_tensor)
        gen_image = []
        for i in range(film_params.shape[0]):
            self.film_siren_nerf.set_film_params(film_params[i])
            gen_image.append(self.renderer(self.film_siren_nerf))
        gen_image = torch.stack(gen_image)
        gen_image = gen_image.permute(0, 3, 1, 2).contiguous()
        return gen_image

    def get_mapping(self, input_tensor):
        film_params = self.mapping_network(input_tensor)
        return film_params

    def set_film_params(self, film_params):
        self.film_siren_nerf.set_film_params(film_params)

    def set_resolution(self, resolution):
        self.renderer.width = resolution
        self.renderer.height = resolution

    def render(self, theta=None, phi=None):
        return self.renderer(self.film_siren_nerf, theta, phi)


"""=============== DISCRIMINATOR ==============="""

# CoordConv is taken from
# https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py

class AddCoords(torch.nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = torch.nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class DownsampleResBlock(torch.nn.Module):
    """The block in discriminator"""
    def __init__(self, input_dim, output_dim):
        super(DownsampleResBlock, self).__init__()
        self.res_layer = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.layer = torch.nn.Sequential(
            CoordConv(input_dim, output_dim, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),
            CoordConv(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.activation = torch.nn.LeakyReLU(0.2)
        self.downsample = torch.nn.AvgPool2d(2)

    def forward(self, input_tensor):
        res = self.res_layer(input_tensor)
        h = self.layer(input_tensor)
        h = h + res
        h = self.activation(h)
        output_tensor = self.downsample(h)
        return output_tensor


class Discriminator(torch.nn.Module):
    def __init__(self, resolution):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.progression_layers = torch.nn.ModuleList([
            DownsampleResBlock(64, 128),    # 64 -> 32
            DownsampleResBlock(128, 256),   # 32 -> 16
            DownsampleResBlock(256, 400),   # 16 -> 8
            DownsampleResBlock(400, 400),   # 8 -> 4
            DownsampleResBlock(400, 400),   # 4 -> 2
        ])
        self.output_layer = torch.nn.Conv2d(400, 1, kernel_size=2)
        self.adapter_layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=1), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(torch.nn.Conv2d(3, 128, kernel_size=1), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(torch.nn.Conv2d(3, 256, kernel_size=1), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(torch.nn.Conv2d(3, 400, kernel_size=1), torch.nn.LeakyReLU(0.2)),
            torch.nn.Sequential(torch.nn.Conv2d(3, 400, kernel_size=1), torch.nn.LeakyReLU(0.2))
        ])
        self.n_layers = len(self.progression_layers)

    def set_resolution(self, resolution):
        self.resolution = resolution

    def forward(self, input_tensor, resolution=None, alpha=-1):
        if resolution is None:
            resolution = self.resolution
        step = self.n_layers - int(np.log2(resolution)) + 1
        h = self.adapter_layers[step](input_tensor)
        for i in range(step, self.n_layers):
            h = self.progression_layers[i](h)
            # fade in
            if i == step and 0 <= alpha < 1:
                skip_h = torch.nn.functional.avg_pool2d(input_tensor, 2)
                skip_h = self.adapter_layers[step + 1](skip_h)
                h = (1 - alpha) * skip_h + alpha * h
        output_tensor = self.output_layer(h).squeeze()
        return output_tensor








