import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from utils.crane_dataset import CraneDataset


class PressureDataset(Dataset):
    def __init__(self, pressure_data: np.array, seq_len: int = 1, pred_distance: int = 1):
        self.__seq_len = seq_len
        self.__pred_distance = pred_distance
        self.__dataset_tensor = torch.FloatTensor(pressure_data)

    def __len__(self):
        return self.__dataset_tensor.shape[0] - (self.__seq_len+self.__pred_distance)

    def __getitem__(self, idx):
        return (self.__dataset_tensor[idx:idx+self.__seq_len],
                self.__dataset_tensor[idx+self.__seq_len+self.__pred_distance])


class PressureDataModule(pl.LightningDataModule):
    def __init__(self, data_path_list: list, seq_len: int = 1, pred_distance: int = 1, batch_size: int = 32,
                 n_of_worker: int = 1):
        super().__init__()

        self.__crane_dataset_inst = CraneDataset(data_path_list)
        self.__train_data = self.__crane_dataset_inst.get_train_dataset()[self.__crane_dataset_inst.get_data_target_names()]
        self.__train_data = self.__train_data.to_numpy()
        self.__train_data = self.__train_data.T.flatten()
        self.__train_data = self.__train_data[::10]
        #self.__train_data = (500000-self.__train_data)/500000


        self.__val_data = self.__crane_dataset_inst.get_val_dataset()[self.__crane_dataset_inst.get_data_target_names()]
        self.__val_data = self.__val_data.to_numpy()
        self.__val_data = self.__val_data.T.flatten()
        self.__val_data = self.__val_data[::10]
        #self.__val_data = (500000-self.__val_data)/500000

        self.__seq_len = seq_len
        self.__pred_distance = pred_distance
        self.__batch_size = batch_size
        self.__n_of_worker = n_of_worker

        self.__train_dataset = None
        self.__val_dataset = None
        self.__test_dataset = None

    def setup(self, stage=None):
        self.__train_dataset = PressureDataset(pressure_data=self.__train_data, seq_len=self.__seq_len,
                                               pred_distance=self.__pred_distance)

        self.__val_dataset = PressureDataset(pressure_data=self.__val_data, seq_len=self.__seq_len,
                                              pred_distance=self.__pred_distance)

    def train_dataloader(self):
        return DataLoader(dataset=self.__train_dataset, batch_size=self.__batch_size, shuffle=True, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.__val_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.__val_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)