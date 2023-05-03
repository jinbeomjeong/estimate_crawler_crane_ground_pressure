import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score


class GroundPressureModel(pl.LightningModule):
    def __init__(self, n_input, n_h1, n_h2, n_output):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.h1_neuron = int(round(n_h1))
        self.h2_neuron = int(round(n_h2))

        self.input_layer = [nn.Linear(n_input, self.h1_neuron), nn.ReLU()]
        self.hidden_layer_1 = [nn.Linear(self.h1_neuron, self.h2_neuron), nn.ReLU()]
        self.output_layer = [nn.Linear(self.h2_neuron, n_output)]

        self.model = nn.Sequential(*self.input_layer, *self.hidden_layer_1, *self.output_layer)

        self.loss_func = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_pred = self.model(x)
        r2_value = r2_score(y.cpu().detach().numpy(), val_pred.cpu().detach().numpy())
        self.log("r2_score", r2_value)

        return r2_value

    # def test_step(self, batch, batch_idx):
    # x, y = batch
    # test_pred = self.model(x)
    # test_loss = F.cross_entropy(test_pred, y)
    # self.log("test_loss", test_loss)

    # return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

        return optimizer
