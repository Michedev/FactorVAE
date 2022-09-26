from itertools import chain

import pytorch_lightning as pl
from functools import partial
import torch
from torch import nn
from torch import distributions

class FactorVAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 opt_vae: partial, opt_discriminator: partial, d: int, gamma: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self._opt_vae = opt_vae
        self._opt_discriminator = opt_discriminator
        self.d = d  # permutation rounds
        self.mse = nn.MSELoss()
        self.gamma = gamma
        self.prior = distribution.Normal(0, 1)

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
        vae_result = self(x1)
        z = vae_result['z']
        d_output: torch.Tensor = self.discriminator(z)
        log_d_output = d_output.log()
        reconstruction_loss = self.mse(x, vae_result['x_hat'])  # note: authors use negative crossentropy here
        loss1 = reconstruction_loss - distributions.kl_divergence(distributions.Normal(vae_result['post_mu'], vae_result['post_std']), self.prior).log() \
                - self.gamma * log_d_output / (1 - d_output)
        loss1 = loss1.sum() / x1.shape[0]
        vae_result = self(x2)
        z_hat = vae_result['z']
        z_perm = self.permute(z_hat)
        d_output_2 = self.discriminator(z_perm)

        loss2 = log_d_output + (1 - d_output_2).log()
        loss2 = loss2.sum() / x.shape[0]
        return [loss1, loss2]

    def configure_optimizers(self):
        opt1 = self._opt_vae(params=chain(self.encoder.parameters(), self.decoder.parameters()))
        opt2 = self._opt_discriminator(params=self.discriminator)
        return [opt1, opt2]

    def permute(self, z: torch.Tensor):
        latent_size: int = z.shape[-1]
        pi = torch.arange(latent_size)
        for _ in range(self.d):
            torch.random.shuffle(pi)
