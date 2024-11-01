import os.path, time
import onnxruntime as ort
import numpy as np

from utils.Dataset import CraneDataset, create_lstm_dataset
from tqdm.auto import tqdm

model_set = []
for i in tqdm(range(10), desc='Loading best models'):
    model_set.append(ort.InferenceSession(os.path.join('models', f'onnx_model_{i}_10.onnx')))

data_file_path_list = []
data_file_name_list = os.listdir('data')

for file_name in data_file_name_list:
    data_file_path_list.append(os.path.join('data', file_name))

dataset_inst = CraneDataset(data_file_path_list)
raw_dataset = dataset_inst.get_dataset()

feature_names = dataset_inst.get_data_feature_names()
target_names = dataset_inst.get_data_target_names()

for t_name in target_names:
    raw_dataset[t_name] = raw_dataset[t_name] / 1000

for t_name in target_names:
    raw_dataset = raw_dataset[raw_dataset[t_name] > 1]

for name in feature_names[3:6]:
    raw_dataset[name] = raw_dataset[name] * 1000

raw_dataset = raw_dataset[raw_dataset['Time(sec)'] > 4]

raw_dataset = raw_dataset[::10]
raw_dataset.reset_index(drop=True, inplace=True)

grad_boom_angle = np.gradient(raw_dataset['Boom_Angle(deg)'])
grad_swing_angle = np.gradient(raw_dataset['Swing_Angle(deg)'])
grad_roll_angle = np.gradient(raw_dataset['Roll_Angle(deg)'])
grad_pitch_angle = np.gradient(raw_dataset['Pitch_Angle(deg)'])

extra_feature_name_list = ['grad_boom_angle', 'grad_swing_angle', 'grad_roll_angle', 'grad_pitch_angle']
new_feature_names = feature_names + extra_feature_name_list

for extra_feature_name, extra_feature in zip(extra_feature_name_list,
                                             [grad_boom_angle, grad_swing_angle, grad_roll_angle, grad_pitch_angle]):
    raw_dataset[extra_feature_name] = extra_feature

train_dataset_1 = raw_dataset[raw_dataset['dataset_type'] == 0]
train_dataset = train_dataset_1.drop(columns=['Time(sec)', 'file_idx', 'safe_state', 'dataset_type'])
train_dataset = train_dataset[new_feature_names[0:4] + new_feature_names[5:10] + target_names]

val_dataset_1 = raw_dataset[raw_dataset['dataset_type'] == 1]
val_dataset = val_dataset_1.drop(columns=['Time(sec)', 'file_idx', 'safe_state', 'dataset_type'])
val_dataset = val_dataset[new_feature_names[0:4] + new_feature_names[5:10] + target_names]


val_feature, val_target = create_lstm_dataset(val_dataset.to_numpy(), seq_len=30, pred_distance=10,
                                              target_idx_pos=9 + 0)
val_feature = val_feature.astype(np.float32)
val_target = val_target.astype(np.float32)

time_list = []

for input_data in tqdm(val_feature):
    t0 = time.time()

    for model in model_set:
        pred = model.run(output_names=None, input_feed={'input': np.expand_dims(input_data, axis=0)})[0][0]
    time_list.append(time.time() - t0)

print('Average inference time per data sequence: {:.3f} seconds'.format(np.mean(time_list)))
