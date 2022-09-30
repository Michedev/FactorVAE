from torch import nn


def dsprites_encoder(input_channels: int, latent_size: int):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=4, stride=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=4, stride=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.GroupNorm(1, 64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2),
        nn.GroupNorm(1, 64),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, latent_size * 2),
    )


def dsprites_decoder(latent_size: int, output_channels: int):
    return nn.Sequential(
        nn.Linear(latent_size, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(),
        nn.Unflatten(1, (64, 4, 4)),
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(1, 64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
    )