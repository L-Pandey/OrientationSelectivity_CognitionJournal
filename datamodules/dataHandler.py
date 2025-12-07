# ImageFolder is a torchvision class used to create a dataset with images and labels.
# This returns a dataset which need to be wrapped with a Dataloader.


import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from PIL import Image
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

class OrientationRecognition(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,
            batch_size,
            pin_memory,
            num_workers,
            val_split,
            shuffle,
            drop_last,
            train_dir,
            img_res,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_split = val_split
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.test_paths = []
        self.train_dir = train_dir
        self.img_res = img_res


    def prepare_data(self,):
        # prepare train and test paths
        #self.train_path = self.data_dir+'train'
        if self.train_dir is None:
            self.train_path = self.data_dir+'train'
        else:
            self.train_path = self.data_dir+self.train_dir
        self.test_path = self.data_dir+'test'
        print("training path - ", self.train_path, " and testing path - ", self.test_path)

    
    def create_splits(self,dataset_size):
        train_ratio = 1-self.val_split
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        return train_size, val_size
        
    # pytorch lightning function
    # add stage condition to load data as per the trainer status
    def setup(self, stage=None):
        # create transform
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        if stage == 'fit' or stage is None:
            dataset = ImageFolder(self.train_path, train_transforms)
            
            train_size, val_size = self.create_splits(len(dataset))
            self.dataset_train, self.dataset_val = random_split(dataset, [train_size, val_size])
        
        if stage == 'test' or stage is None:
            self.dataset_test = ImageFolder(self.test_path, val_transforms)
            print("len of dataset test - ", len(self.dataset_test))

        print(
              "len of dataset train - ", len(self.dataset_train), 
              ", len of dataset val - ", len(self.dataset_val)
              )

    # custom function - to test the model on individual classes 
    def get_individual_test_dirs(self,):
        return ImageFolder(self.test_path)


    def default_transforms(self) -> Callable:
        # transforms the image to tensor
        #return T.ToTensor()
        return T.Compose([
        T.Resize((self.img_res, self.img_res)),
        T.ToTensor()
    ])

    # pytorch lightning function
    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)


    # pytorch lightning function
    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_val)


    # pytorch lightning function
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_test)


    # custom function
    def _data_loader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )