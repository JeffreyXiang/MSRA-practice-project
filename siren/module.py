import torch
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


class Siren(torch.nn.Linear):
    """Siren layer: Linear layer and sine activation"""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize a Dense layer

        :param input_dim: input dimension
        :param output_dim: output dimension
        :param activation: non-linear activation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Siren, self).__init__(input_dim, output_dim)

    def forward(self, input_tensor):
        return torch.sin(super(Siren, self).forward(input_tensor))

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, -np.sqrt(6 / self.input_dim) / 30, np.sqrt(6 / self.input_dim) / 30)
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


class SirenMLP(torch.nn.Module):
    """Multi Layer Perceptron using Siren"""
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super(SirenMLP, self).__init__()
        self.input_layer = Siren(input_dim, hidden_dim)
        self.hidden_layers = [Siren(hidden_dim, hidden_dim) for i in range(hidden_layers)]
        self.output_layer = Dense(hidden_dim, output_dim, activation='sigmoid')
        torch.nn.init.uniform_(self.input_layer.weight, -1 / input_dim, 1 / input_dim)

    def forward(self, input_tensor):
        h = self.input_layer(input_tensor)
        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h)
        output_tensor = self.output_layer(h)
        return output_tensor


class ReLUMLP(torch.nn.Module):
    """Multi Layer Perceptron using Siren"""
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers):
        super(ReLUMLP, self).__init__()
        self.input_layer = Dense(input_dim, hidden_dim, activation='relu')
        self.hidden_layers = [Dense(hidden_dim, hidden_dim, activation='relu') for i in range(hidden_layers)]
        self.output_layer = Dense(hidden_dim, output_dim, activation='sigmoid')

    def forward(self, input_tensor):
        h = self.input_layer(input_tensor)
        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h)
        output_tensor = self.output_layer(h)
        return output_tensor

