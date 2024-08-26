import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import pandas as pd


class PressureModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 16, num_layers: int = 1, learning_rate=0.003):
        super(PressureModel, self).__init__()

        self.lr = learning_rate
        self.__model = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.mse_loss_fc = nn.MSELoss(reduction='mean')
        self.mae_loss_fc = nn.L1Loss(reduction='mean')
        self.__test_data = list()
        self.__test_pred = list()

        self.__test_label = list()
        self.__test_pred = list()

    def forward(self, input_data):
        input_data = input_data.unsqueeze(dim=2)  # input feature is 1
        out, _ = self.__model(input_data)
        out = self.fc(out[:, -1])

        return out.squeeze()

    def training_step(self, batch, batch_idx):
        data, label = batch

        pred = self.forward(data)
        mse_loss = self.mse_loss_fc(pred, label)
        mae_loss = self.mse_loss_fc(pred, label)

        self.log(name='train_loss', value=mse_loss, sync_dist=True, prog_bar=True)
        self.log(name='pressure_loss', value=mae_loss, sync_dist=True, prog_bar=False)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)

        mse_loss = self.mse_loss_fc(pred, label)
        mae_loss = self.mae_loss_fc(pred, label)

        self.log(name='val_loss', value=mse_loss, sync_dist=True, prog_bar=True)
        self.log(name='pressure_loss', value=mae_loss, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred = self.forward(data)

        self.__test_label.append(label.cpu().numpy())
        self.__test_pred.append(pred.cpu().numpy())

        print(label.cpu().numpy()[0], pred.cpu().numpy()[0])
        #print('---')
        #self.log(name='test_loss', value=mse_loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        #self.log(name='pressure_loss', value=self.mae_loss_fc(pred, label), on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        #self.log(name='pressure_gt', value=label_arr, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)
        #self.log(name='pressure_pred', value=pred_arr, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

    def on_test_epoch_end(self):
        df = pd.DataFrame(np.stack([np.hstack(self.__test_pred), np.hstack(self.__test_label)], axis=1), columns=['pred', 'label'])
        df.to_csv(path_or_buf='test_data.csv', index=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)