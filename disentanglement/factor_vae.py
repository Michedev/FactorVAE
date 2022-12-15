import pytorch_lightning as pl
from random import randint, random
import torch
from tqdm import tqdm

def disentangle(model: pl.LightningModule, rounds: int, dataset_size: int):
    """
    Factor VAE disentanglement metric implementation
    """
    correct_guesses = 0
    for _ in tqdm(range(rounds)):
        y = randint(0, model.latent_size - 1)
        fixed_value = random() * 6 - 3
        z = torch.randn((dataset_size, model.latent_size), device=model.device)
        z[:, y] = fixed_value
        x_gen = model.generate(batch_size=dataset_size, z=z)

        z_hat: torch.Tensor = model.forward_encoder(x_gen)['z']
        z_hat_norm = z / z_hat.std()
        var_z_hat_norm = z_hat_norm.var(dim=0)
        y_hat = var_z_hat_norm.argmin().item()
        correct_guesses += int(y == y_hat)
    return correct_guesses / rounds
