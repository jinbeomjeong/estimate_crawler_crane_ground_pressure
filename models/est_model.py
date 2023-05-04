import torch
import torch.nn as nn
import pytorch_lightning as pl
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score

class BasicBlock(pl.LightningModule):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.n_neuron = n_input*n_output
        self.basic_block = nn.Sequential(nn.Linear(self.n_neuron, self.n_neuron), nn.Linear(self.n_neuron, self.n_neuron))
        self.shortcut_block = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.basic_block(x) + self.shortcut_block(x))


class GroundPressureModel(pl.LightningModule):
    def __init__(self, n_input: int, n_layer: int, n_output: int):
        super().__init__()
        block = BasicBlock(n_input, n_output)
        self.loss = nn.MSELoss()
        self.metric = R2Score()
        self.layer = [block]*n_layer
        self.model = nn.Sequential(nn.Linear(n_input, n_input*n_output), *self.layer, nn.Linear(n_input*n_output, n_output))
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss_val = self.loss(y_hat, y)
        self.log('train_loss', loss_val)
        print(r2_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()))
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
