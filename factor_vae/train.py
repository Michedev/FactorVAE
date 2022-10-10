import hydra
import pkg_resources
from omegaconf import DictConfig, OmegaConf
from path import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import omegaconf
from factor_vae.utils.paths import CODE_MODEL, ROOT


@hydra.main(pkg_resources.resource_filename("factor_vae", 'config'), 'train.yaml')
def train(config: DictConfig):
    ckpt = None

    pl.seed_everything(config.seed)
    if config.ckpt is not None:
        ckpt = ROOT / Path(config.ckpt)
        assert ckpt.exists() and ckpt.isdir(), f"Checkpoint {ckpt} does not exist or is not a directory"
        config = OmegaConf.load(ckpt / 'config.yaml')
        os.chdir(ckpt.abspath())
        print('loaded original config from', ckpt / 'config.yaml')
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f, resolve=True)
    print(omegaconf.OmegaConf.to_yaml(config))

    model: pl.LightningModule = hydra.utils.instantiate(config.model, gradient_clip_val=config.gradient_clip_val,
                                                        gradient_clip_algorithm=config.gradient_clip_algorithm)
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset.train, train=True)
    val_dataset: Dataset = hydra.utils.instantiate(config.dataset.val, train=False)
    config.batch_size = config.batch_size * 2  # each training step requires two batches as specified in the paper
    print('double batch size to value ', config.batch_size)

    if not Path('model').exists():
        CODE_MODEL.copytree('model')  # copy source code of model under experiment directory

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=pin_memory, shuffle=True)
    ckpt_callback = ModelCheckpoint('./', 'best',
                                    monitor='valid/loss_vae_epoch',
                                    auto_insert_metric_name=False, save_last=True,
                                    every_n_train_steps=config.model.log_freq)
    print_stats_image(train_dataset)

    callbacks = [ckpt_callback]
    if config.enable_beta_warmup:
        callbacks.append(hydra.utils.instantiate(config.beta_warmup))
    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         resume_from_checkpoint=ckpt / 'best.ckpt' if ckpt is not None else None,
                         max_steps=config.max_steps,
                         max_epochs=config.max_epochs)
    print(model)
    trainer.fit(model, train_dl, val_dl)


def print_stats_image(train_dataset):
    first_image = train_dataset[0]['image']
    print(f"First image shape: {first_image.shape}")
    print(f"First image type: {first_image.dtype}")
    print('First image min/max: {}/{}'.format(first_image.min(), first_image.max()))
    print('First image mean/std: {}/{}'.format(first_image.mean(), first_image.std()))
    print(f"First image: {first_image}")


if __name__ == '__main__':
    print('pythonpath:', os.getenv('PYTHONPATH'))
    train()
