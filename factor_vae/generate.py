from random import randint

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from path import Path
import torchvision
import torch

from factor_vae.utils.paths import CONFIG, ROOT


@hydra.main(CONFIG, 'generate.yaml')
def main(config: DictConfig):
    ckpt_folder = ROOT / Path(config.checkpoint_path)
    assert ckpt_folder.exists() and ckpt_folder.isdir(), f"Checkpoint {ckpt_folder} does not exist or is not a directory"
    assert (ckpt_folder / 'config.yaml').exists(), f"Checkpoint {ckpt_folder} does not contain a config.yaml file"
    assert (ckpt_folder / 'best.ckpt').exists(), f"Checkpoint {ckpt_folder} does not contain a best.ckpt file"

    pl.seed_everything(config.seed)

    ckpt_config = OmegaConf.load(ckpt_folder / 'config.yaml')

    model: pl.LightningModule = hydra.utils.instantiate(ckpt_config.model)
    print('loading', ckpt_folder / 'best.ckpt')
    model.load_state_dict(torch.load(ckpt_folder / 'best.ckpt', map_location=config.device)['state_dict'])

    model.eval()
    model.freeze()

    generated_batch = model.generate(config.batch_size)
    print('generated batch', generated_batch.shape)
    img_grid = torchvision.utils.make_grid(generated_batch, nrow=config.grid_rows)
    torchvision.utils.save_image(img_grid, ckpt_folder / 'generated.png')
    print('saved', ckpt_folder / 'generated.png')


if __name__ == '__main__':
    main()
