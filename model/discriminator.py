from torch import nn


def make_sequential_discriminator(input_size: int, num_hidden_layers: int, hidden_size: int) -> nn.Module:
    """
    Creates a discriminator MLP network with activation function LeakyReLU.
    The output layer is a sigmoid layer with output size 1.
    :param input_size: the input size
    :param num_hidden_layers: the number of hidden layers
    :param hidden_size: the size of each hidden layer
    :return: the discriminator network
    """
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.LeakyReLU(negative_slope=0.2))
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
    layers.append(nn.Linear(hidden_size, 2))
    return nn.Sequential(*layers)
