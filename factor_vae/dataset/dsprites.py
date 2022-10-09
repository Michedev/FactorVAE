import os
import subprocess

from torch.utils.data import Dataset
from factor_vae.utils.paths import DATA, ROOT
import h5py
import torch
from path import Path
import os
import numpy as np


class DSpritesImages(Dataset):
    """
    DSprites dataset containing only images
    """

    features_metadata = {
        'color': {
            'type': 'categorical',
            'num_classes': 1,
            'values': ['white'],
            'index': 0
        },
        'shape': {
            'type': 'categorical',
            'num_classes': 3,
            'values': ['square', 'ellipse', 'heart'],
            'index': 1
        },
        'scale': {
            'type': 'numerical',
            'index': 2,
        },
        'orientation': {
            'type': 'numerical',
            'index': 3,
        },
        'posX': {
            'type': 'numerical',
            'index': 4,
        },
        'posY': {
            'type': 'numerical',
            'index': 5,
        }
    }

    dsprites_images = None
    dsprites_features = None

    def __init__(self, train_size: float, train: bool = True, download=True,
                 dataset_len=737280, load_features: bool = False, width: int = 64, height: int = 64, input_channels: int = 1):
        assert 737280 >= dataset_len > 0, 'dataset_len must be in range [1, 737280]'
        self.train_size = train_size
        self.train = train
        self.dataset_len = dataset_len
        self.load_features = load_features
        self.width = width
        self.height = height
        self.input_channels = input_channels
        dsprites_path = Path(os.getenv('DSPRITES', default=None)) or DATA / 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
        if not dsprites_path.exists():
            if download:
                subprocess.call(['pipenv', 'run', 'download-dsprites'], cwd=ROOT)
                if not dsprites_path.exists():
                    raise RuntimeError('Download failed')
            raise FileNotFoundError('Please download the dataset from {} and place it in {}'
                                    .format('https://github.com/deepmind/dsprites-dataset', DATA))
        self.train_len = int(self.dataset_len * train_size)
        self.val_len = self.dataset_len - self.train_len

        self.dataset_len = self.train_len if train else self.val_len

        # load hdf5 file
        self.dsprites = h5py.File(dsprites_path, 'r')
        if self.dsprites_images is None:
            print('Preloading images')
            DSpritesImages.dsprites_images = self.dsprites['imgs'][:]
            DSpritesImages.i_rand = np.random.permutation(len(DSpritesImages.dsprites_images))
            DSpritesImages.dsprites_images = DSpritesImages.dsprites_images[DSpritesImages.i_rand]
            print('Done')
        self.dsprites_images = DSpritesImages.dsprites_images
        print(f'{len(DSpritesImages.dsprites_images)=}')
        self.dsprites_features = None
        if load_features:
            if self.dsprites_features is None:
                print('Preloading features')
                DSpritesImages.dsprites_features = self.dsprites['latents']['values'][:]
                DSpritesImages.dsprites_features = DSpritesImages.dsprites_features[DSpritesImages.i_rand]
                print('Done')
            print(f'{len(DSpritesImages.dsprites_features)=}')
            self.dsprites_features = DSpritesImages.dsprites_features
        if self.train:
            self.dsprites_images = self.dsprites_images[:self.train_len]
            if load_features:
                self.dsprites_features = self.dsprites_features[:self.train_len]
        else:
            self.dsprites_images = self.dsprites_images[self.train_len:]
            if load_features:
                self.dsprites_features = self.dsprites_features[self.train_len:]
        print('Done')

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i) -> dict:
        image = torch.tensor(self.dsprites_images[i]).float()
        image = image.view(1, 64, 64)
        result = dict(image=image)
        if self.load_features:
            for feature, metadata in self.features_metadata.items():
                result[feature] = self.dsprites_features[i, metadata['index']]
                if metadata['type'] == 'categorical':
                    result[feature] = torch.tensor(result[feature], dtype=torch.long) - 1
                else:
                    result[feature] = torch.tensor(result[feature], dtype=torch.float)
        return result


if __name__ == '__main__':
    dsprites_path = os.getenv('DSPRITES', default=None) or DATA / 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
    dsprites_path = Path(dsprites_path)
    dsprites = h5py.File(dsprites_path, 'r')
    print(dsprites.keys())
    print(dsprites['imgs'].shape)
    print(dsprites['latents']['classes'].shape)
    print(dsprites['latents']['values'].shape)
    print('First row', dsprites['latents']['classes'][0])
    print('First row', dsprites['latents']['values'][0])
    for feature, metadata in DSpritesImages.features_metadata.items():
        i = metadata["index"]
        values = dsprites['latents']['values'][:, i]
        if metadata["type"] == "categorical":
            print(f"possible values of {i}th feature {feature}", set(values))
        else:
            print(f'average of {i}th feature {feature}', np.mean(values), '+-', np.std(values))
