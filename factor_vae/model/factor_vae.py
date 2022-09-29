from functools import partial
from itertools import chain

import pytorch_lightning as pl
import tensorguard as tg
import torch
from torch import distributions
from torch import nn


class FactorVAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 opt_vae: partial, opt_discriminator: partial, d: int, gamma: float,
                 latent_size: int, log_freq: int):
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
        self.log_freq = log_freq
        self.iteration = 0
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.mse = nn.MSELoss(reduction='sum')
        self.beta = 1.0

    def forward(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.exp(0.5 * post_logvar)
        z = distributions.Normal(post_mu, post_std).rsample()
        x_hat = self.decoder(z)
        tg.guard(x_hat, '*, 1, W, H')
        return dict(z=z, x_hat=x_hat, post_mu=post_mu, post_std=post_std)

    def forward_encoder(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.exp(0.5 * post_logvar)
        z = distributions.Normal(post_mu, post_std).rsample()
        return dict(z=z, post_mu=post_mu, post_std=post_std)

    def training_step(self, batch, batch_idx):

        x = batch['image']
        x1, x2 = x.chunk(2, dim=0)  # each training step requires two batches as specified in the paper
        tg.guard(x1, '*, 1, W, H')
        tg.guard(x2, '*, 1, W, H')
        opt_vae, opt_d = self.optimizers()

        vae_result_x1 = self(x1)
        loss_vae = self.calc_loss_vae(vae_result_x1, x1)

        opt_vae.zero_grad()
        self.manual_backward(loss_vae)
        opt_vae.step()

        vae_result_x2 = self.forward_encoder(x2)
        loss_discriminator = self.calc_discriminator_loss(vae_result_x2)

        opt_d.zero_grad()
        self.manual_backward(loss_discriminator)
        opt_d.step()

        with torch.no_grad():
            loss = loss_vae + loss_discriminator
        if self.iteration % self.log_freq == 0:
            self.log('train/loss_vae', loss_vae, prog_bar=True)
            self.log('train/loss_discriminator', loss_discriminator, prog_bar=True)
            self.log('train/loss', loss)
        self.iteration += 1
        return dict(loss=loss, loss_vae=loss_vae, loss_discriminator=loss_discriminator)

    def validation_step(self, batch, batch_idx) -> dict:
        x = batch['image']
        x1, x2 = x.chunk(2, dim=0)  # each training step requires two batches as specified in the paper
        tg.guard(x1, '*, 1, W, H')
        tg.guard(x2, '*, 1, W, H')

        vae_result_x1 = self(x1)
        loss_vae = self.calc_loss_vae(vae_result_x1, x1)
        vae_result_x2 = self.forward_encoder(x2)

        loss_discriminator = self.calc_discriminator_loss(vae_result_x2)

        loss = loss_vae + loss_discriminator
        self.log('valid/loss_vae', loss_vae, prog_bar=True, on_step=True, on_epoch=True)
        self.log('valid/loss_discriminator', loss_discriminator, prog_bar=True, on_step=True, on_epoch=True)
        self.log('valid/loss', loss, on_step=True, on_epoch=True)
        return dict(loss=loss, loss_vae=loss_vae, loss_discriminator=loss_discriminator)

    def calc_discriminator_loss(self, vae_result_x2):
        z = vae_result_x2['z'].detach()
        z_perm = torch.clone(z)
        z_perm = self.permute(z_perm).detach()
        d_z = self.discriminator(z).log_softmax(dim=-1)
        d_z_perm = self.discriminator(z_perm).log_softmax(dim=-1)
        loss_discriminator = d_z[:, 1] + d_z_perm[:, 0]
        loss_discriminator = loss_discriminator.sum() / (2 * z.shape[0])
        return loss_discriminator

    def calc_loss_vae(self, vae_result_x1, x1):
        """
        Compute the loss for the VAE part of the model. The loss is composed by the standard vAE loss,
        i.e. reconstruction loss and the KL, plus the discriminator loss.
        :param vae_result_x1: dict of VAE results on first half of the batch
        :param x1: input images
        :return: vae scalar loss
        """
        z = vae_result_x1['z']
        d_z: torch.Tensor = self.discriminator(z)
        neg_log_reconstruction_loss = self.bce(vae_result_x1['x_hat'],
                                               x1)  # note: authors use negative crossentropy here
        post_z = distributions.Normal(vae_result_x1['post_mu'], vae_result_x1['post_std'])
        kl = distributions.kl_divergence(post_z, self.prior).sum()
        density_ratio = (d_z[:, 1] - d_z[:, 0]).sum()
        loss_vae = neg_log_reconstruction_loss - self.beta * kl + self.gamma * (d_z[:, 0] - d_z[:, 1]).sum()
        loss_vae = loss_vae / x1.shape[0]
        if self.iteration % self.log_freq == 0:
            self.log('train/recon_loss', neg_log_reconstruction_loss / x1.shape[0])
            self.log('train/kl_loss', -  kl / x1.shape[0])
            self.log('train/density_ratio_loss', density_ratio / x1.shape[0])
        return loss_vae

    def configure_optimizers(self):
        opt1 = self._opt_vae(params=chain(self.encoder.parameters(), self.decoder.parameters()))
        opt2 = self._opt_discriminator(params=self.discriminator.parameters())
        return [opt1, opt2]

    def permute(self, z: torch.Tensor):
        latent_size: int = z.shape[-1]
        for _ in range(self.d):
            pi = torch.randperm(latent_size)
            z = z[:, pi]
        return z

    def generate(self, batch_size: int = 1, z: torch.Tensor = None):
        if z is None:
            z = self.prior.sample((batch_size, self.latent_size)).squeeze(-1)
        assert len(z.shape) == 2, 'z must be a 2D tensor - current shape: {}'.format(z.shape)
        return self.decoder(z)
