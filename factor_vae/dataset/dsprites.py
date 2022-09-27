import os
import subprocess

from torch.utils.data import Dataset
from factor_vae.utils.paths import DATA, ROOT
import h5py
import torch


# todo: loading time is very slow, check new alternative or set better caching for h5py
class DSpritesImages(Dataset):
    """
    DSprites dataset containing only images
    """

    def __init__(self, train_size: float, train: bool = True, download=True, preload=True):
        self.train_size = train_size
        self.train = train
        dsprites_path = DATA / 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
        if not dsprites_path.exists():
            if download:
                subprocess.call(['poe', 'download-dsprites'], cwd=ROOT)
                if not dsprites_path.exists():
                    raise RuntimeError('Download failed')
            raise FileNotFoundError('Please download the dataset from {} and place it in {}'
                                    .format('https://github.com/deepmind/dsprites-dataset', DATA))
        self.dsprites_len = 737280  # took from github repo
        self.train_len = int(self.dsprites_len * train_size)
        self.val_len = self.dsprites_len - self.train_len

        self.dataset_len = self.train_len if train else self.val_len
        # load hdf5 file
        self.dsprites = h5py.File(dsprites_path, 'r')['imgs']
        if preload:
            if self.train:
                self.dsprites = self.dsprites[:self.train_len]
            else:
                self.dsprites = self.dsprites[self.train_len:]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i) -> dict:
        image = torch.tensor(self.dsprites[i]).float()
        image = image.view(1, 64, 64)
        return dict(image=image)

# TODO: implement dataset with all the features
