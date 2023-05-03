import torch, os, json
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

with open('.' + os.sep + os.path.join('models', 'params_dnn_20220207-012403.json'), 'r') as file:
    hyper_params = json.load(file)

n_input = hyper_params['n_of_inputs']
h1_neuron = hyper_params['h1_neuron']
h2_neuron = hyper_params['h2_neuron']
n_output = hyper_params['n_of_outputs']


class GroundPressureModel(pl.LightningModule):
    def __init__(self, n_input, n_h1, n_h2, n_output):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.h1_neuron = int(round(n_h1))
        self.h2_neuron = int(round(n_h2))

        self.input_layer = [nn.Linear(n_input, h1_neuron), nn.ReLU()]
        self.hidden_layer = [nn.Linear(h1_neuron, h2_neuron), nn.ReLU()]
        self.output_layer = [nn.Linear(h2_neuron, n_output)]

        self.model = nn.Sequential(*self.input_layer, *self.hidden_layer, *self.output_layer)

        self.loss_func = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log('train_loss', loss)

        return loss

    # def validation_step(self, batch, batch_idx):
    # x, y = batch
    # val_pred = self.model(x)
    # val_loss = F.cross_entropy(val_pred, y)
    # self.log("train_loss", val_loss)
    # return val_loss

    # def test_step(self, batch, batch_idx):
    # x, y = batch
    # test_pred = self.model(x)
    # test_loss = F.cross_entropy(test_pred, y)
    # self.log("test_loss", test_loss)

    # return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

        return optimizer


model = GroundPressureModel(n_input, h1_neuron, h2_neuron, n_output)


gpu = torch.device('cpu')
model.load_state_dict(torch.load('./models/est_ground_pressure.pt', map_location=gpu))
model.eval()

input_a = torch.tensor(np.array([1.5, 5, 30, 20], dtype=np.float32), device=gpu)
print(model(input_a))

