import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn import preprocessing


class PressureModel(pl.LightningModule):
    def __init__(self, seq_len: int = 1, hidden_size: int = 16, num_layers: int = 1, batch_size = 2, learning_rate=0.01):
        super(PressureModel, self).__init__()

        self.__lr = learning_rate
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        self.__batch_size = batch_size

        self.__model = nn.LSTM(input_size=seq_len, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.__fc = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        #self.__batch_norm = nn.BatchNorm1d(hidden_size)

        self.mse_loss_fc = nn.MSELoss(reduction='mean')
        self.mae_loss_fc = nn.L1Loss(reduction='mean')

        self.__test_data = list()
        self.__test_pred = list()

        self.__test_label = list()

        self.__val_pred = None
        self.__val_label = None

        self.__h0 = nn.Parameter(torch.zeros(self.__num_layers, self.__batch_size, self.__hidden_size))
        self.__c0 = nn.Parameter(torch.zeros(self.__num_layers, self.__batch_size, self.__hidden_size))

    def forward(self, input_data):
        #h0 = self.__h0.clone().detach().to(input_data.device)
        #c0 = self.__c0.clone().detach().to(input_data.device)
        out, _ = self.__model(input_data)
        out = out[:, -1, :]
        #out = self.__batch_norm(out)
        out = self.__fc(out)

        return out

    def training_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)
        self.log(name='train_loss', value=mse_loss, prog_bar=True)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)

        mse_loss = self.mse_loss_fc(pred, label)
        mae_loss = self.mae_loss_fc(pred, label)

        self.log(name='val_loss', value=mse_loss, prog_bar=True)
        self.log(name='pressure_loss', value=mae_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)

        self.__test_label.append(label.cpu().numpy())
        self.__test_pred.append(pred.cpu().numpy())

        print(label.cpu().numpy()[0], pred.cpu().numpy()[0])


    def on_test_epoch_end(self):
        df = pd.DataFrame(np.stack([np.hstack(self.__test_pred), np.hstack(self.__test_label)], axis=1), columns=['pred', 'label'])
        df.to_csv(path_or_buf='test_data.csv', index=False)

    def configure_optimizers(self):
        return optim.Adam(self.__model.parameters(), lr=self.__lr)


class LoadEstModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 16, num_layers: int = 1, learning_rate=0.003):
        super(LoadEstModel, self).__init__()

        self.__lr = learning_rate
        self.__model = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.__fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.__mse_loss_fc = nn.MSELoss(reduction='mean')
        self.__mae_loss_fc = nn.L1Loss(reduction='mean')
        self.__test_data = list()
        self.__test_pred = list()

    def forward(self, input_data):
        input_data = input_data.unsqueeze(dim=2)  # input feature is 1
        out, _ = self.__model(input_data)
        out = self.__fc(out[:, -1])

        return out

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.forward(data)
        mse_loss = self.__mse_loss_fc(pred.squeeze(), label)
        mae_loss = self.__mae_loss_fc(pred.squeeze(), label)

        self.log(name='train_loss', value=mse_loss, sync_dist=True, prog_bar=True)
        self.log(name='load_loss(N)', value=mae_loss, sync_dist=True, prog_bar=False)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)

        mse_loss = self.__mse_loss_fc(pred, label)
        mae_loss = self.__mae_loss_fc(pred, label)

        self.log(name='val_loss', value=mse_loss, sync_dist=True, prog_bar=True)
        self.log(name='val_load_loss(N)', value=mae_loss, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)

        data_arr = data.cpu().numpy()
        pred_arr = pred.cpu().numpy()
        label_arr = label.cpu().numpy()

        self.log(name='test_loss', value=mse_loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log(name='pressure_loss', value=self.mae_loss_fc(pred, label), on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log(name='pressure_gt', value=label_arr[0], on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log(name='pressure_gt2', value=data_arr[0, -1], on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        self.log(name='pressure_pred', value=pred_arr[0], on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

    def on_train_epoch_end(self):
        for name, param in self.__model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and torch.all(param.grad == 0):
                    print(f'Gradient is zero for {name}')

    # def on_test_epoch_end(self):
    #     df = pd.DataFrame(np.vstack(self.__test_data), columns=['time', 'pressure_measurement'])
    #     df.to_csv(path_or_buf='test_data.csv', index=False)
    #
    #     df = pd.DataFrame(np.array(self.__test_pred), columns=['pressure_prediction'])
    #     df.to_csv(path_or_buf='test_pred.csv', index=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.__lr)