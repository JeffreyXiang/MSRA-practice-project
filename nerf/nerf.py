import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np

class Dense(torch.nn.Linear):
    """Dense layer: Linear layer and an activation"""
    def __init__(self, input_dim: int, output_dim: int, activation: str = "linear") -> None:
        """
        Initialize a Dense layer

        :param input_dim: input dimension
        :param output_dim: output dimension
        :param activation: non-linear activation
        """
        self.activation_name = activation
        self.activation = eval('torch.' + activation) if activation != 'linear' else None
        super(Dense, self).__init__(input_dim, output_dim)

    def forward(self, input_tensor):
        if self.activation is not None:
            return self.activation(super(Dense, self).forward(input_tensor))
        else:
            return super(Dense, self).forward(input_tensor)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation_name))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class PositionalEncoding:
    """sine and cosine positional encoder"""
    def __init__(self, input_dim, length):
        """
        Initialize Positional Encoding module

        :param input_dim: input dimension
        :param length: encode length (output dimension is double of it)
        """
        self.input_dim = input_dim
        self.length = length
        self.output_dim = input_dim * length * 2

    def __call__(self, input_tensor):
        output = []
        for i in range(self.length):
            output.append(torch.sin(2.0**i * input_tensor))
            output.append(torch.cos(2.0**i * input_tensor))
        return torch.cat(output, -1)


class NeRF(torch.nn.Module):
    """Neural Radiance Field model"""
    def __init__(self):
        super(NeRF, self).__init__()
        self.pe_pos = PositionalEncoding(3, 10)
        self.pe_dir = PositionalEncoding(3, 4)
        self.layers_pos = torch.nn.ModuleList([
            Dense(60, 256, activation='relu'),
            Dense(256, 256, activation='relu'),
            Dense(256, 256, activation='relu'),
            Dense(256, 256, activation='relu'),
            Dense(256, 256, activation='relu'),
            Dense(256 + 60, 256, activation='relu'),
            Dense(256, 256, activation='relu'),
            Dense(256, 256, activation='relu')
        ])
        self.layers_dir = torch.nn.ModuleList([
            Dense(256, 256, activation='linear'),
            Dense(256 + 24, 128, activation='relu'),
        ])
        self.output_layer_sigma = Dense(256, 1, activation='relu')
        self.output_layer_rgb = Dense(128, 3, activation='sigmoid')

    def forward(self, input_tensor):
        input_pos, input_dir = torch.split(input_tensor, [3, 3], dim=-1)
        embedded_pos = self.pe_pos(input_pos)
        embedded_dir = self.pe_dir(input_dir)
        h = self.layers_pos[0](embedded_pos)
        h = self.layers_pos[1](h)
        h = self.layers_pos[2](h)
        h = self.layers_pos[3](h)
        h = self.layers_pos[4](h)
        h = torch.cat([embedded_pos, h], -1)
        h = self.layers_pos[5](h)
        h = self.layers_pos[6](h)
        h = self.layers_pos[7](h)
        sigma = self.output_layer_sigma(h)
        h = self.layers_dir[0](h)
        h = torch.cat([h, embedded_dir], -1)
        h = self.layers_dir[1](h)
        rgb = self.output_layer_rgb(h)
        outputs = torch.cat([rgb, sigma], -1)
        return outputs


class Siren(torch.nn.Linear):
    """Siren layer: Linear layer and sine activation"""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize a Siren layer

        :param input_dim: input dimension
        :param output_dim: output dimension
        :param activation: non-linear activation
        """
        super(Siren, self).__init__(input_dim, output_dim)

    def forward(self, input_tensor):
        return torch.sin(super(Siren, self).forward(input_tensor))

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, -np.sqrt(6 / self.input_dim) / 30, np.sqrt(6 / self.input_dim) / 30)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class SirenNeRF(torch.nn.Module):
    """Neural Radiance Field model"""
    def __init__(self):
        super(SirenNeRF, self).__init__()
        self.layers_pos = torch.nn.ModuleList([
            Siren(3, 256),
            Siren(256, 256),
            Siren(256, 256),
            Siren(256, 256),
            Siren(256, 256),
            Siren(256 + 3, 256),
            Siren(256, 256),
            Siren(256, 256)
        ])
        self.layers_dir = torch.nn.ModuleList([
            Dense(256, 256, activation='linear'),
            Siren(256 + 3, 128),
        ])
        self.output_layer_sigma = Dense(256, 1, activation='relu')
        self.output_layer_rgb = Dense(128, 3, activation='sigmoid')

    def forward(self, input_tensor):
        input_pos, input_dir = torch.split(input_tensor, [3, 3], dim=-1)
        h = self.layers_pos[0](input_pos)
        h = self.layers_pos[1](h)
        h = self.layers_pos[2](h)
        h = self.layers_pos[3](h)
        h = self.layers_pos[4](h)
        h = torch.cat([input_pos, h], -1)
        h = self.layers_pos[5](h)
        h = self.layers_pos[6](h)
        h = self.layers_pos[7](h)
        sigma = self.output_layer_sigma(h)
        h = self.layers_dir[0](h)
        h = torch.cat([h, input_dir], -1)
        h = self.layers_dir[1](h)
        rgb = self.output_layer_rgb(h)
        outputs = torch.cat([rgb, sigma], -1)
        return outputs

