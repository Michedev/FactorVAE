from itertools import chain
from math import prod

import pytorch_lightning as pl
from functools import partial
import torch
from torch import nn
from torch import distributions
import omegaconf

omegaconf.OmegaConf.register_new_resolver('sum', lambda *x: sum(float(el) for el in x))
omegaconf.OmegaConf.register_new_resolver('prod', lambda *x: prod(float(el) for el in x))

class FactorVAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 opt_vae: partial, opt_discriminator: partial, d: int, gamma: float,
                 latent_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self._opt_vae = opt_vae
        self._opt_discriminator = opt_discriminator
        self.d = d  # permutation rounds
        self.mse = nn.MSELoss()
        self.gamma = gamma
        self.latent_size = latent_size
        self.prior = distributions.Normal(0, 1)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.exp(0.5 * post_logvar)
        z = post_mu + post_std * torch.randn_like(post_logvar)
        x_hat = self.decoder(z)
        return dict(z=z, x_hat=x_hat, post_mu=post_mu, post_std=post_std)

    def forward_encoder(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.exp(0.5 * post_logvar)
        z = post_mu + post_std * torch.randn_like(post_logvar)
        return dict(z=z, post_mu=post_mu, post_std=post_std)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x1, x2 = x.chunk(2, dim=0)  # each training step requires two batches as specified in the paper
        opt_vae, opt_d = self.optimizers()
        opt_vae.zero_grad()
        opt_d.zero_grad()
        vae_result = self(x1)
        z = vae_result['z']
        d_output: torch.Tensor = self.discriminator(z)
        log_d_output = d_output.log()
        reconstruction_loss = self.mse(x, vae_result['x_hat'])  # note: authors use negative crossentropy here
        loss_vae = reconstruction_loss - distributions.kl_divergence(
            distributions.Normal(vae_result['post_mu'], vae_result['post_std']), self.prior).log() \
                - self.gamma * log_d_output / (1 - d_output)
        loss_vae = loss_vae.sum() / x1.shape[0]
        vae_result = self(x2)
        z_hat = vae_result['z']
        z_perm = self.permute(z_hat)
        d_output_2 = self.discriminator(z_perm)

        loss_discriminator = log_d_output + (1 - d_output_2).log()
        loss_discriminator = loss_discriminator.sum() / x.shape[0]
        self.manual_backward(loss_vae)
        opt_vae.step()
        self.manual_backward(loss_discriminator)
        opt_d.step()
        return [loss_vae, loss_discriminator]

    def configure_optimizers(self):
        opt1 = self._opt_vae(params=chain(self.encoder.parameters(), self.decoder.parameters()))
        opt2 = self._opt_discriminator(params=self.discriminator)
        return [opt1, opt2]

    def permute(self, z: torch.Tensor):
        latent_size: int = z.shape[-1]
        for _ in range(self.d):
            pi = torch.randperm(latent_size)
            z = z[:, pi]
        return z
