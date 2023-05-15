import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score


class ResidualBlock(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.active = nn.ReLU()
        self.layers = []

        for i in range(n_hidden):
            self.layers.append(nn.Linear(n_input, n_output))
            self.layers.append(self.active)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.layers(x) + x
        output = self.active(output)

        return output


class ResidualRegression(pl.LightningModule):
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(n_input, n_input*n_output, bias=False))
        self.output_layer = nn.Sequential(nn.Linear(n_input*n_output, n_output, bias=False))
        self.res_block_1 = ResidualBlock(n_input*n_output, n_hidden, n_input*n_output)
        self.res_block_2 = ResidualBlock(n_input*n_output, n_hidden, n_input*n_output)
        self.res_block_3 = ResidualBlock(n_input*n_output, n_hidden, n_input*n_output)
        self.res_block_4 = ResidualBlock(n_input*n_output, n_hidden, n_input*n_output)
        self.res_block_5 = ResidualBlock(n_input*n_output, n_hidden, n_input*n_output)

        self.loss = nn.MSELoss()

    def forward(self, x):
        output = self.input_layer(x)
        output = self.res_block_1(output)
        output = self.res_block_2(output)
        output = self.res_block_3(output)
        output = self.res_block_4(output)
        output = self.res_block_5(output)
        output = self.output_layer(output)

        return output

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
        self.input_layer = nn.Linear(n_input, n_input*n_output)
        self.output_layer = nn.Linear(n_input*n_output, n_output)
        self.active = nn.ReLU()

        self.layers = []
        self.layers.append(self.input_layer)
        self.layers.append(self.active)

        for i in range(n_hidden):
            self.layers.append(nn.Linear(n_input*n_output, n_input*n_output))
            self.layers.append(self.active)

        self.layers.append(self.output_layer)
        self.model = nn.Sequential(*self.layers)
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
