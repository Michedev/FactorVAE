from typing import Iterator, Tuple, overload

import torch
from path import Path
from utils.paths import SAVED_MODELS
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import hydra

from utils.paths import ROOT


def iter_experiments() -> Iterator[Tuple[Path, DictConfig]]:
    for experiment in SAVED_MODELS.dirs():
        config_path = experiment / 'config.yaml'
        config = OmegaConf.load(config_path)
        yield experiment, config


def iter_experiments_with_checkpoint() -> Iterator[Tuple[Path, DictConfig, pl.LightningModule]]:
    for experiment_path, config in iter_experiments():
        model = hydra.utils.instantiate(config.model)
        model.load_from_checkpoint(experiment_path / 'best.ckpt')


def load_checkpoint_model_eval(checkpoint_path: Path, seed: int, device: str) -> dict:
    ckpt_folder = ROOT / Path(checkpoint_path)
    assert ckpt_folder.exists() and ckpt_folder.isdir(), f"Checkpoint {ckpt_folder} does not exist or is not a directory"
    assert (ckpt_folder / 'config.yaml').exists(), f"Checkpoint {ckpt_folder} does not contain a config.yaml file"
    assert (ckpt_folder / 'best.ckpt').exists(), f"Checkpoint {ckpt_folder} does not contain a best.ckpt file"
    pl.seed_everything(seed)
    ckpt_config = OmegaConf.load(ckpt_folder / 'config.yaml')
    model: pl.LightningModule = hydra.utils.instantiate(ckpt_config.model)
    print('loading', ckpt_folder / 'best.ckpt')
    model.load_state_dict(torch.load(ckpt_folder / 'best.ckpt', map_location=device)['state_dict'])
    model = model.eval()
    model.freeze()
    return dict(ckpt_folder=ckpt_folder, ckpt_config=ckpt_config, model=model)
