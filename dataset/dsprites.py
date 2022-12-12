import os
import subprocess
from typing import List, Union, Literal

from torch.utils.data import Dataset
from utils.paths import DATA, ROOT
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

    dsprites_path = Path(os.getenv('DSPRITES', default=None)) or DATA / 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
    dsprites_images = None
    dsprites_features = None

    def __init__(self, splits: List[Union[int, float]],  split: Union[int, Literal['train', 'val', 'test']],
                 load_features: bool = False, width: int = 64, height: int = 64, input_channels: int = 1):
        self._assert_splits(splits)

        assert split in ['train', 'val', 'test'] or isinstance(split, int), split
        split = split if isinstance(split, int) else ['train', 'val', 'test'].index(split)
        assert 0 <= split < len(splits), f'{split=} must be between 0 and {len(splits)=}'


        self.load_features = load_features
        self.width = width
        self.height = height
        self.input_channels = input_channels
        if not self.dsprites_path.exists():
            raise FileNotFoundError('Please download the dataset from {} and place it in {}'
                                    .format('https://github.com/deepmind/dsprites-dataset', DATA))
        index_path = DATA / 'dsprites_index.npy'
        if not index_path.exists():
            print('creating index')
            dsprites_index = np.arange(737280)
            for _ in range(20): np.random.shuffle(dsprites_index)
            np.save(index_path, dsprites_index)
            print('index saved in {}'.format(index_path))
        dsprites_index = np.load(index_path)
        print(f'{splits =}')
        splits.insert(0, 0)
        self.dataset_len = splits[split + 1] - splits[split]

        # load hdf5 file
        self.dsprites = h5py.File(self.dsprites_path, 'r')
        if DSpritesImages.dsprites_images is None:
            print('Preloading images')
            DSpritesImages.dsprites_images = self.dsprites['imgs'][:]
            DSpritesImages.dsprites_images = DSpritesImages.dsprites_images[dsprites_index]
            print('Done')
        self.dsprites_images = DSpritesImages.dsprites_images
        self.dsprites_features = None
        if load_features:
            if self.dsprites_features is None:
                print('Preloading features')
                DSpritesImages.dsprites_features = self.dsprites['latents']['values'][:]
                DSpritesImages.dsprites_features = DSpritesImages.dsprites_features[dsprites_index]
                print('Done')
            print(f'{len(DSpritesImages.dsprites_features)=}')
            self.dsprites_features = DSpritesImages.dsprites_features
        self.dsprites_images = self.dsprites_images[splits[split]:splits[split + 1]]
        if load_features:
            self.dsprites_features = self.dsprites_features[splits[split]:splits[split + 1]]
        print('Done')

    def _assert_splits(self, splits: List[Union[int, float]]):
        assert 1 <= len(splits) <= 3, 'splits must be a list of 1, 2 or 3 elements (train, val, test)'
        if isinstance(splits[0], int):
            assert all(isinstance(split, int) for split in splits), splits
            assert 1 <= sum(splits) <= 737280, f'splits {sum(splits)} must sum to a number between 1 and 737280'
        else:
            # splits are percentages
            assert all(isinstance(split, float) for split in splits), splits
            assert 0 < sum(splits) <= 1, f'splits {sum(splits)} must sum to a number between 0 and 1'
            for i in range(len(splits)):
                if i > 0:
                    splits[i] = int(splits[i] * 737280 + splits[i - 1])
                else:
                    splits[i] = int(splits[i] * 737280)
            splits[-1] = min(splits[-1], 737280)

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

    @classmethod(f)
    def load_random_single_image(cls, splits: List[Union[int, float]], split: Union[int, Literal['train', 'val', 'test']]):
        dsprites = h5py.File(cls.dsprites_path, 'r')
        image = torch.tensor(dsprites['imgs'][index]).float()
        image = image.view(1, 64, 64)
        return image

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
