import math
import os
from typing import List, Union, Dict

import hydra
import torch.nn
import yaml

from utils.experiment_tools import load_checkpoint_model_eval
from utils.paths import CONFIG
from torch import nn
import pytorch_lightning as pl
import json
import numpy as np

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
        if self.global_step % 1000 == 0:
            self.log('dci/train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        yhat = self(x)
        loss = 0.0
        results = dict()
        for feature, metadata in self.features_metadata.items():
            yhat_feature = yhat[:, self.features_slices[feature]]
            y_feature = batch[feature]
            loss = loss + self.loss_dict[feature](yhat_feature, y_feature)
            if metadata['type'] == 'categorical':
                results[f'acc_{feature}'] = (yhat_feature.argmax(dim=1) == y_feature).float().mean()
                self.log(f'dci/val_acc_{feature}', (yhat_feature.argmax(dim=1, keepdim=True) == y_feature).float().mean())
            else:
                results[f'R2_{feature}'] = 1 - (((yhat_feature - y_feature) ** 2).mean() / y_feature.var())
                self.log(f'dci/val_R2_{feature}', 1 - ((yhat_feature - y_feature) ** 2).mean() / y_feature.var())
        loss = loss + \
               self.lambda_1 * self.linear_model.weight.norm(1) + \
               self.lambda_2 * self.linear_model.weight.norm(2)
        results['loss'] = loss
        self.log('dci/val_loss', loss, prog_bar=True)
        return results


    def configure_optimizers(self):
        return torch.optim.Adam(self.linear_model.parameters(), lr=1e-3)


def instantiate_linear_model(latent_size: int, features_metadata: dict):
    output_size = 0
    for feature, metadata in features_metadata.items():
        output_size += metadata['num_classes'] if metadata['type'] == 'categorical' else 1
    return nn.Linear(latent_size, output_size)


@torch.no_grad()
def compute_dci_disentanglement(params: torch.Tensor) -> float:
    """
    >>> abs(compute_dci_disentanglement(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))) <= 1e-5  #tolerance of 1e-5 because log constant
    True


    :param params: Linear model weights matrix
    :return: the dci disentanglement score
    """
    L, M = params.shape
    params = params.abs()
    P = params / params.sum(dim=1, keepdim=True)

    log10_P = torch.log10(P + 1e-5)

    entropy_Pi = (- P * (log10_P / math.log10(M))).sum(dim=1)

    D_i = 1 - entropy_Pi

    phi_i = (params.sum(dim=1) / params.sum())

    return (phi_i * D_i).sum().item()

def compute_dci_completeness(params: torch.Tensor) -> float:
    L, M = params.shape
    params = params.abs()

    R = params / params.sum(dim=0, keepdim=True)
    entropy_Rj = (- R * (torch.log10(R + 1e-5) / math.log10(L))).sum(dim=0)
    C_j = 1 - entropy_Rj

    return C_j.mean().item()

@hydra.main(CONFIG, 'dci.yaml')
def main(config):
    ckpt = load_checkpoint_model_eval(config.checkpoint_path, config.seed, 'cpu')
    model = ckpt['model']
    ckpt_folder = ckpt['ckpt_folder']
    ckpt_config = ckpt['ckpt_config']
    os.chdir(ckpt_folder)
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
    model_ckpt = pl.callbacks.ModelCheckpoint(dci_ckpt, 'best', monitor='dci/val_loss',
                                              mode='min', save_last=True, save_weights_only=True)
    callbacks = [model_ckpt]

    if config.early_stop:
        early_stopping = pl.callbacks.EarlyStopping('dci/val_loss', patience=2, mode='min')
        callbacks.append(early_stopping)
    trainer = pl.Trainer(accelerator=config.accelerator, devices=config.devices,
                         max_epochs=config.max_epochs, callbacks=callbacks)
    if not dci_ckpt.joinpath('last.ckpt').exists():
        trainer.fit(dci_model, train_dl, val_dl)
    else:
        print('Loading checkpoint')
    dci_model.load_state_dict(torch.load(dci_ckpt.joinpath('best.ckpt'))['state_dict'])
    avg_outputs: List[Dict[str, float]] = trainer.validate(dci_model, val_dl, ckpt_path=dci_ckpt / 'best.ckpt')
    assert len(avg_outputs) == 1, 'Only one validation epoch should be run'
    avg_outputs: Dict[str, float] = avg_outputs[0]
    with open(dci_ckpt / 'metrics.yaml', 'w') as f:
        yaml.dump(avg_outputs, f)
    dict_params = dict(weight=dict(), bias=dict())
    print(f'{dci_model.linear_model.weight.shape=}')
    for feature, f_slice in dci_model.features_slices.items():
        w_feature: torch.Tensor = dci_model.linear_model.weight[:, f_slice]
        b_feature = dci_model.linear_model.bias[f_slice]
        dict_params['weight'][feature] = w_feature.tolist()
        dict_params['bias'][feature] = b_feature.tolist()
    with open(dci_ckpt / 'params.json', 'w') as f:
        json.dump(dict_params, f)
    save_dci_values_(avg_outputs, dci_ckpt, dci_model)
    print('Saved metrics and params in', dci_ckpt)


def save_dci_values_(avg_outputs, dci_ckpt, dci_model):
    disentanglement = compute_dci_disentanglement(dci_model.linear_model.weight)
    completeness = compute_dci_completeness(dci_model.linear_model.weight)
    informativeness = np.mean([v for k, v in avg_outputs.items() if 'R2' in k or 'acc' in k]).item()
    global_dci = dict(disentanglement=disentanglement, completeness=completeness, informativeness=informativeness)
    dci = {'global': global_dci, 'per_feature': dict()}
    for feature, metadata in dci_model.features_metadata.items():
        f_slice = dci_model.features_slices[feature]
        f_params = dci_model.linear_model.weight[f_slice, :]
        f_disentanglement = compute_dci_disentanglement(f_params)
        f_informativeness = [v for k, v in avg_outputs.items() if feature in k]
        assert len(f_informativeness) == 1, f_informativeness
        f_informativeness = f_informativeness[0]
        dci['per_feature'][feature] = dict(disentanglement=f_disentanglement,
                                           informativeness=f_informativeness)

    with open(dci_ckpt / 'dci.yaml', 'w') as f:
        yaml.dump(dci, f)



if __name__ == '__main__':
    main()
