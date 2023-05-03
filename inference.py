import torch, os, json
import numpy as np
from models.est_model import GroundPressureModel


with open('.' + os.sep + os.path.join('models', 'params_dnn_20220207-012403.json'), 'r') as file:
    hyper_params = json.load(file)

n_input = hyper_params['n_of_inputs']
h1_neuron = hyper_params['h1_neuron']
h2_neuron = hyper_params['h2_neuron']
n_output = hyper_params['n_of_outputs']


model = GroundPressureModel(n_input, h1_neuron, h2_neuron, n_output)


gpu = torch.device('cpu')
model.load_state_dict(torch.load('./models/est_ground_pressure.pt', map_location=gpu))
model.eval()

input_a = torch.tensor(np.array([1, 1, 40, 20], dtype=np.float32), device=gpu)
print(model(input_a).detach().numpy())

