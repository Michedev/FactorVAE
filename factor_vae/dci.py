import hydra
import torch.nn

from utils.experiment_tools import load_checkpoint_model_eval
from utils.paths import CONFIG
from torch import nn
import pytorch_lightning as pl


def instantiate_loss_dict(features_metadata):
    return {feature: nn.CrossEntropyLoss() if metadata['type'] == 'categorical' else nn.MSELoss()
            for feature, metadata in features_metadata.items()}


def instantiate_features_slices(features_metadata):
    i = 0
    slices = dict()
    for feature, metadata in features_metadata.items():
        if metadata['type'] == 'categorical':
            f_slice = slice(i, i + metadata['num_classes'])
        else:
            f_slice = slice(i, i + 1)
        i = f_slice.stop
        slices[feature] = f_slice
    return slices


class DCILinear(pl.LightningModule):

    def __init__(self, vae: torch.nn.Module, latent_size: int, features_metadata: dict, lambda_1: float,
                 lambda_2: float):
        super().__init__()
        self.vae = vae
        self.latent_size = latent_size
        self.features_metadata = features_metadata
        self.linear_model = instantiate_linear_model(vae.latent_size, features_metadata)
        self.loss_dict = instantiate_loss_dict(features_metadata)
        self.features_slices = instantiate_features_slices(features_metadata)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, x):
        with torch.no_grad():
            z = self.vae.forward_encoder(x)['z']
        return self.linear_model(z)

    def training_step(self, batch, batch_idx):
        x = batch['image']
        yhat = self(x)
        loss = 0.0
        for feature, metadata in self.features_metadata.items():
            loss = loss + self.loss_dict[feature](yhat[:, self.features_slices[feature]], batch[feature])
        loss = loss + \
               self.lambda_1 * self.linear_model.weight.norm(1) + \
               self.lambda_2 * self.linear_model.weight.norm(2)
        self.log('dci/train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        yhat = self(x)
        loss = 0.0
        for feature, metadata in self.features_metadata.items():
            yhat_feature = yhat[:, self.features_slices[feature]]
            y_feature = batch[feature]
            loss = loss + self.loss_dict[feature](yhat_feature, y_feature)
            if metadata['type'] == 'categorical':
                self.log(f'dci/val_acc_{feature}', (yhat_feature.argmax(dim=1, keepdim=True) == y_feature).float().mean())
            else:
                self.log(f'dci/val_R2_{feature}', 1 - ((yhat_feature - y_feature) ** 2).mean() / y_feature.var())
        loss = loss + \
               self.lambda_1 * self.linear_model.weight.norm(1) + \
               self.lambda_2 * self.linear_model.weight.norm(2)
        self.log('dci/val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.linear_model.parameters(), lr=1e-3)


def instantiate_linear_model(latent_size: int, features_metadata: dict):
    output_size = 0
    for feature, metadata in features_metadata.items():
        output_size += metadata['num_classes'] if metadata['type'] == 'categorical' else 1
    return nn.Linear(latent_size, output_size)


@hydra.main(CONFIG, 'dci.yaml')
def main(config):
    ckpt = load_checkpoint_model_eval(config.checkpoint_path, config.seed, config.device)
    model = ckpt['model']
    ckpt_folder = ckpt['ckpt_folder']
    ckpt_config = ckpt['ckpt_config']
    dci_ckpt = ckpt_folder / 'dci'
    if not dci_ckpt.exists():
        dci_ckpt.mkdir()
    ckpt_config['dataset']['load_features'] = True
    train_dataset = hydra.utils.instantiate(ckpt_config.dataset, train=True)
    val_dataset = hydra.utils.instantiate(ckpt_config.dataset, train=False)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    dci_model = DCILinear(model, ckpt_config.model.latent_size, train_dataset.features_metadata,
                          config.lambda_1, config.lambda_2)
    model_ckpt = pl.callbacks.ModelCheckpoint(dci_ckpt / 'best.ckpt', monitor='dci/val_loss',
                                              mode='min', save_last=True)

    trainer = pl.Trainer(accelerator='cpu', max_epochs=config.max_epochs, callbacks=[model_ckpt])

    trainer.fit(dci_model, train_dl, val_dl)


if __name__ == '__main__':
    main()
