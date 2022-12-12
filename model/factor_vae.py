from functools import partial
from itertools import chain

import pytorch_lightning as pl
import tensorguard as tg
import torch
import torchvision
from torch import distributions
from torch import nn


class FactorVAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator: nn.Module,
                 opt_vae: partial, opt_discriminator: partial, d: int, gamma: float,
                 latent_size: int, log_freq: int, gradient_clip_val: float = 0.0,
                 gradient_clip_algorithm: str = 'norm', debug: bool = False):
        super().__init__()
        assert gradient_clip_algorithm in ['norm', 'value']
        assert gradient_clip_val >= 0.0
        assert latent_size >= d, 'latent_size must be greater than or equal to d (shuffled latent dimensions)'

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
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self._clip_grad_ = None
        self.debug = debug
        if self.gradient_clip_val > 1e-6:  # clip gradients
            clip_grad = getattr(torch.nn.utils, f'clip_grad_{self.gradient_clip_algorithm}_')
            self._clip_grad_ = lambda p: clip_grad(p, self.gradient_clip_val)

        tg.set_dim('L', self.latent_size)

    def forward(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.exp(0.5 * post_logvar)
        z = distributions.Normal(post_mu, post_std).rsample()
        x_hat = self.decoder(z)
        return dict(z=z, x_hat=x_hat, post_mu=post_mu, post_std=post_std)

    def forward_encoder(self, x: torch.Tensor):
        post_mu, post_logvar = self.encoder(x).chunk(2, dim=-1)
        post_std = torch.nn.functional.softplus(post_logvar)
        z = distributions.Normal(post_mu, post_std).rsample()
        return dict(z=z, post_mu=post_mu, post_std=post_std)

    def training_step(self, batch, batch_idx):

        x = batch['image']
        x1, x2 = x.chunk(2, dim=0)  # each training step requires two batches as specified in the paper

        opt_vae, opt_d = self.optimizers()

        vae_result_x1 = self(x1)
        loss_vae = self.calc_loss_vae(vae_result_x1, x1)

        opt_vae.zero_grad()
        self.manual_backward(loss_vae)
        if self._clip_grad_ is not None: self._clip_grad_(self._vae_parameters())
        opt_vae.step()

        vae_result_x2 = self.forward_encoder(x2)
        loss_discriminator = self.calc_discriminator_loss(vae_result_x2)

        opt_d.zero_grad()
        self.manual_backward(loss_discriminator)
        if self._clip_grad_ is not None: self._clip_grad_(self.discriminator.parameters())
        opt_d.step()

        with torch.no_grad():
            loss = loss_vae + loss_discriminator
        if self.iteration % self.log_freq == 0:
            self.log('train/loss_vae', loss_vae, prog_bar=True)
            self.log('train/loss_discriminator', loss_discriminator, prog_bar=True)
            self.log('train/loss', loss)
            self.log('train/x1_d_z0', vae_result_x1['z'][:, 0].detach().mean(dim=0).item(), )
            self.log('train/x1_d_z1', vae_result_x1['z'][:, 1].detach().mean(dim=0).item(), )
            self.log('train/x2_d_z0', vae_result_x2['z'][:, 0].detach().mean(dim=0).item(), )
            self.log('train/x2_d_z1', vae_result_x2['z'][:, 1].detach().mean(dim=0).item(), )

        self.iteration += 1
        return dict(loss=loss, loss_vae=loss_vae, loss_discriminator=loss_discriminator)

    def _vae_parameters(self):
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def validation_step(self, batch, batch_idx) -> dict:
        if batch_idx == 0:
            batch_gen_images = self.generate(64)
            grid_images = torchvision.utils.make_grid(batch_gen_images, nrow=8)
            self.logger.experiment.add_image('generated_images_epoch', grid_images, self.current_epoch)
        x = batch['image']
        x1, x2 = x.chunk(2, dim=0)  # each training step requires two batches as specified in the paper
        tg.guard(x1, '*, 1, W, H')
        tg.guard(x2, '*, 1, W, H')

        vae_result_x1 = self(x1)

        tg.guard(vae_result_x1['x_hat'], '*, 1, W, H')
        tg.guard(vae_result_x1['z'], '*, L')

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
        if self.debug:
            print('d_z', d_z)
            print('d_z_perm', d_z_perm)
        loss_discriminator = - d_z[:, 0] - d_z_perm[:, 1]
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
        density_ratio = (d_z[:, 0] - d_z[:, 1]).sum()
        loss_vae = neg_log_reconstruction_loss + self.beta * kl + self.gamma * density_ratio
        loss_vae = loss_vae / x1.shape[0]
        if self.debug:
            print('neg_log_reconstruction_loss', neg_log_reconstruction_loss)
            print('kl', kl)
            print('density_ratio', density_ratio)
            print('loss_vae', loss_vae)
            print('d_z', d_z)
        if self.iteration % self.log_freq == 0:
            self.log('train/recon_loss', neg_log_reconstruction_loss / x1.shape[0])
            self.log('train/kl_loss', self.beta * kl / x1.shape[0])
            self.log('train/density_ratio_loss', self.gamma * density_ratio / x1.shape[0])
        return loss_vae

    def configure_optimizers(self):
        opt1 = self._opt_vae(params=self._vae_parameters())
        opt2 = self._opt_discriminator(params=self.discriminator.parameters())
        return [opt1, opt2]

    def permute(self, z: torch.Tensor):
        bs, latent_size = z.shape
        for i in range(self.d):
            pi = torch.randperm(bs)
            z[:, i] = z[pi, i]
        return z

    def generate(self, batch_size: int = 1, z: torch.Tensor = None):
        if z is None:
            z = self.prior.sample((batch_size, self.latent_size)).squeeze(-1).to(self.device)
            print('sample from prior with shape', z.shape)
        assert len(z.shape) == 2, 'z must be a 2D tensor - current shape: {}'.format(z.shape)
        return self.decoder(z).sigmoid()
