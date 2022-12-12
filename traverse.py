from random import randint

import hydra
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
from path import Path
import torchvision
import torch

from utils.paths import CONFIG, ROOT
from utils.experiment_tools import load_checkpoint_model_eval


def plot_traverse(model, config, z_original, ckpt_folder):
    fig, axs = plt.subplots(model.latent_size, config.z_linspace.num, figsize=(config.z_linspace.num, model.latent_size))
    for i in range(model.latent_size):
        for j, z_i in enumerate(torch.linspace(config.z_linspace.min, config.z_linspace.max, config.z_linspace.num)):
            z = torch.clone(z_original)
            z[:, i] = z_i
            generated_img = model.generate(batch_size=1, z=z)
            axs[i, j].imshow(generated_img.squeeze(0).permute(1, 2, 0), cmap='gray')
            # disable ax ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == (model.latent_size-1):
                axs[i, j].set_xlabel(f'{z_i:.2f}')
            if j == 0:
                axs[i, j].set_ylabel(f'$z_{i+1}$')
    fig.tight_layout()
    fig.savefig(ckpt_folder / 'traverse.png')
    print('saved', ckpt_folder / 'traverse.png')
    

@hydra.main(CONFIG, 'traverse.yaml')
def main(config: DictConfig):
    ckpt = load_checkpoint_model_eval(ROOT / config.checkpoint_path, config.seed, config.device)
    model = ckpt['model']
    ckpt_folder = ckpt['ckpt_folder']
    ckpt_config = ckpt['ckpt_config']

    # load random image from test set
    dataset_class = hydra.utils._locate(path=ckpt_config.dataset['_target_'])
    img = dataset_class.load_random_single_image_from_config(ckpt_config.dataset, split='test').unsqueeze(0)

    # get latent vector
    z_original = model.forward_encoder(img)['post_mu']

    plot_traverse(model, config, z_original, ckpt_folder)

    
if __name__ == '__main__':
    main()