import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score


class ResidualBlock(pl.LightningModule):
    def __init__(self, n_input: int, n_hidden: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_input, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, n_hidden)
        self.layer_3 = nn.Linear(n_hidden, n_input)
        self.active = nn.ReLU()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.active(out)
        out = self.layer_2(out)
        out = self.active(out)
        out = self.layer_3(out) + x
        out = self.active(out)

        return out


class ResidualRegression(pl.LightningModule):
    def __init__(self, n_input: int, n_layer: int, n_output: int):
        super().__init__()

        self.layers = []
        for i in range(n_layer):
            self.layers.append(ResidualBlock(n_input, n_input*n_output*3))

        self.output_layer = nn.Linear(n_input, n_output)
        self.layers.append(self.output_layer)

        self.model = nn.Sequential(*self.layers)
        self.loss = nn.MSELoss()
        print(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss_val = self.loss(y_hat, y)
        self.log('train_loss', loss_val)
        # print(r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))
        return loss_val

    '''
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.metric.update(y_hat, y)
        score = self.metric.compute()
        self.log('val_r2score', score.item())
    '''

    # def test_step(self, batch, batch_idx):
    # x, y = batch
    # test_pred = self.model(x)
    # test_loss = F.cross_entropy(test_pred, y)
    # self.log("test_loss", test_loss)

    # return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

        return optimizer


class DNNRegression(pl.LightningModule):
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.layer_1 = nn.Linear(n_input, n_hidden)
        self.layer_2 = nn.Linear(n_hidden, n_hidden)
        self.layer_3 = nn.Linear(n_hidden, n_output)
        self.active = nn.ReLU()
        self.model = nn.Sequential(self.layer_1, self.active, self.layer_2, self.active, self.layer_3)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss_val = self.loss(y_hat, y)
        self.log('train_loss', loss_val)
        return loss_val

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
