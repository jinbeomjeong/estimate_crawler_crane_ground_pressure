#import torch
import numpy as np
#import pytorch_lightning as pl
#from torch.utils.data import DataLoader, Dataset
from utils.Dataset import CraneDataset
from sklearn import preprocessing


class LoadDatasetV1(Dataset):
    def __init__(self, data: np.array, seq_len: int = 1, pred_distance: int = 1):
        self.__seq_len = seq_len
        self.__pred_distance = pred_distance
        self.__dataset_tensor = torch.FloatTensor(data)

    def __len__(self):
        return self.__dataset_tensor.shape[0] - (self.__seq_len+self.__pred_distance)

    def __getitem__(self, idx):
        return (self.__dataset_tensor[idx:idx+self.__seq_len],
                self.__dataset_tensor[idx+self.__seq_len+self.__pred_distance])


class PressureDataset(Dataset):
    def __init__(self, data: np.array, seq_len: int = 1, pred_distance: int = 1):
        load_pos = 0
        load_pos += 3

        n_sample = data.shape[0] - (seq_len + pred_distance)

        self.__feature_arr = np.zeros(shape=[n_sample, 4, seq_len], dtype=np.float32)
        self.__target_arr = np.zeros(shape=[n_sample, 1], dtype=np.float32)

        for i in range(n_sample):
            self.__feature_arr[i] = data[i:i+seq_len, :].T
            self.__target_arr[i] = data[i+seq_len+pred_distance, load_pos]

    def __len__(self):
        return self.__feature_arr.shape[0]

    def __getitem__(self, idx):

        return torch.FloatTensor(self.__feature_arr[idx]), torch.FloatTensor(self.__target_arr[idx])


class PressureDataModule(pl.LightningDataModule):
    def __init__(self, train_data_path_list: list, val_data_path_list: list, seq_len: int = 1, pred_distance: int = 1,
                 batch_size: int = 32, n_of_worker: int = 1):
        super().__init__()

        self.__scaler = preprocessing.StandardScaler()

        self.__train_crane_dataset_inst = CraneDataset(train_data_path_list)
        self.__train_data = self.__train_crane_dataset_inst.get_dataset()[self.__train_crane_dataset_inst.get_data_target_names()[0]]
        self.__train_data = self.__train_data.to_numpy()
        self.__train_data = self.__train_data.T.flatten()
        #self.__train_data = self.__scaler.fit_transform(self.__train_data.reshape(-1, 1))
        #self.__train_data = self.__train_data.squeeze()

        self.__train_data = self.__train_data[::10]
        #self.__train_data = (500000-self.__train_data)/500000

        self.__val_crane_dataset_inst = CraneDataset(val_data_path_list)
        self.__val_data = self.__val_crane_dataset_inst.get_dataset()[self.__val_crane_dataset_inst.get_data_target_names()[0]]
        self.__val_data = self.__val_data.to_numpy()
        self.__val_data = self.__val_data.T.flatten()
        #self.__val_data = self.__scaler.transform(self.__val_data.reshape(-1, 1))
        #self.__val_data = self.__val_data.squeeze()

        self.__val_data = self.__val_data[::10]
        #self.__val_data = (500000-self.__val_data)/500000

        self.__seq_len = seq_len
        self.__pred_distance = pred_distance
        self.__batch_size = batch_size
        self.__n_of_worker = n_of_worker


    def setup(self, stage=None):
        self.__train_dataset = LoadDatasetV1(data=self.__train_data, seq_len=self.__seq_len, pred_distance=self.__pred_distance)

        self.__val_dataset = LoadDatasetV1(data=self.__val_data, seq_len=self.__seq_len, pred_distance=self.__pred_distance)

    def train_dataloader(self):
        return DataLoader(dataset=self.__train_dataset, batch_size=self.__batch_size, shuffle=True, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.__val_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.__val_dataset, batch_size=self.__batch_size, shuffle=False, num_workers=self.__n_of_worker,
                          persistent_workers=True)

    def get_scaler(self):
        return self.__scaler