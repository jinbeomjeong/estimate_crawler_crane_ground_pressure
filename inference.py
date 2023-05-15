import torch, os, json
import numpy as np
import pandas as pd
import seaborn as sns
from models.est_model import ResidualRegression, DNNRegression
from sklearn.metrics import mean_absolute_error

cpu = torch.device('cpu')

ref_data = pd.read_csv('./resources/sim_data_edit.csv')
feature_names = ['lift_weight(ton)', 'lift_height(m)', 'rising_angle(deg)', 'swing_angle(deg)']
left_target_names = ['left-0.0m', 'left-0.675m', 'left-1.35m', 'left-2.025m', 'left-2.7m', 'left-3.375m', 'left-4.05m', 'left-4.725m', 'left-5.4m', 'left-6.075m', 'left-6.75m']
right_target_names = ['right-0.0m', 'right-0.675m', 'right-1.35m', 'right-2.025m', 'right-2.7m', 'right-3.375m', 'right-4.05m', 'right-4.725m', 'right-5.4m', 'right-6.075m', 'right-6.75m']
target_names = left_target_names + right_target_names

feature = ref_data[feature_names]
target = ref_data[target_names].values

input_T = torch.FloatTensor(feature.values, device=cpu)

error_abs_arr = np.zeros(target.shape)
error_rel_arr = np.zeros(target.shape)

with open('.' + os.sep + os.path.join('models', 'params_dnn_20220207-012403.json'), 'r') as file:
    hyper_params = json.load(file)

n_inputs = hyper_params['n_of_inputs']
n_outputs = hyper_params['n_of_outputs']
n_layers = hyper_params['n_of_hidden']

model = ResidualRegression(n_inputs, n_layers, n_outputs)

model.load_state_dict(torch.load('./models/est_ground_pressure.pt', map_location=cpu))
model.eval()

prd = model(input_T).detach().numpy()

for i in range(target.shape[0]):
    for j in range(target.shape[1]):
        error_abs_arr[i, j] = np.abs(target[i, j] - prd[i, j])

        if target[i, j] == 0:
            error_abs_arr[i, j] = 0
        else:
            error_rel_arr[i, j] = np.abs(target[i, j] - prd[i, j]) / target[i, j]


target_1d = target.reshape(-1, 1)
prd_1d = prd.reshape(-1, 1)

print(error_abs_arr)
print(error_rel_arr)
