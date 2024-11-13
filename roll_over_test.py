import os, joblib
import numpy as np

from utils.Dataset import CraneDataset


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

# for t_name in target_names:
#     raw_dataset = raw_dataset[raw_dataset[t_name]>=0]

raw_dataset = raw_dataset[raw_dataset['Time(sec)'] > 5]

raw_dataset.reset_index(drop=True, inplace=True)

left_load_name_list = ['Actual_Load_Left_1(N)', 'Actual_Load_Left_2(N)', 'Actual_Load_Left_3(N)', 'Actual_Load_Left_4(N)', 'Actual_Load_Left_5(N)']
right_load_name_list = ['Actual_Load_Right_1(N)', 'Actual_Load_Right_2(N)', 'Actual_Load_Right_3(N)', 'Actual_Load_Right_4(N)', 'Actual_Load_Right_5(N)']

data_sample = raw_dataset[raw_dataset['file_idx']==11]
time_arr = data_sample['Time(sec)'].to_numpy()
angle_arr = data_sample['Swing_Angle(deg)'].to_numpy()


angle_list_1 = [0, 45, 90, 135, 180, 225, 270, 315, 360]
left_ref_load_list = [0.50, 0.70, 0.80, 0.70, 0.50, 0.30, 0.20, 0.30, 0.50]
right_ref_load_list = [0.50, 0.30, 0.20, 0.30, 0.50, 0.70, 0.80, 0.70, 0.50]

angle_list_2 = [0, 45, 90, 135, 180, 225, 270, 315, 360]
front_ref_load_list = [0.6, 0.9, 1.0, 0.9, 0.6, 0.9, 1.0, 0.9, 0.6]

ref_angle_arr = np.linspace(0, 360, 36001)

ref_left_load_ratio_2 = 0.5*np.sin(np.deg2rad(ref_angle_arr))+0.5
ref_left_load_ratio_2 = (0.6*ref_left_load_ratio_2)+0.2

ref_right_load_ratio_2 = 1 - ref_left_load_ratio_2

ref_front_load_ratio = 0.3 * np.sin(np.radians(ref_angle_arr) + np.pi / 2) + 0.5
ref_rear_load_ratio = 1 - ref_front_load_ratio

ref_left_load_ratio = np.interp(ref_angle_arr, angle_list_1, left_ref_load_list)
ref_right_load_ratio = np.interp(ref_angle_arr, angle_list_1, right_ref_load_list)

margin_arr = np.zeros(4)
load_ratio_arr = np.zeros(4)

def calculate_margin(ref_left_ratio, ref_right_ratio, ref_front_ratio, ref_rear_ratio, ground_load, swing_angle_i):
    global margin_arr, load_ratio_arr

    left_load_idx_list = [0, 1, 2, 3, 4]
    right_load_idx_list = [5, 6, 7, 8, 9]

    front_load_idx_list = [0, 1, 5, 6]
    rear_load_idx_list = [3, 4, 8, 9]

    load_total = np.sum(ground_load)
    left_load_total = np.sum(ground_load[left_load_idx_list])
    right_load_total = np.sum(ground_load[right_load_idx_list])

    front_view_load_total = np.sum(ground_load[front_load_idx_list +rear_load_idx_list])
    front_load_total = np.sum(ground_load[front_load_idx_list])
    rear_load_total = np.sum(ground_load[rear_load_idx_list])

    load_ratio_arr[0] = (left_load_total / load_total)
    load_ratio_arr[1] = (right_load_total / load_total)
    load_ratio_arr[2] = (front_load_total / front_view_load_total)
    load_ratio_arr[3] = (rear_load_total / front_view_load_total)

    angle_idx = np.argmin(np.abs(ref_angle_arr - swing_angle_i))

    if 0 <= swing_angle_i < 180:
        margin_arr[0] = ref_left_ratio[angle_idx] - load_ratio_arr[0]
        margin_arr[1] = load_ratio_arr[1] - ref_right_ratio[angle_idx]

    if 180 <= swing_angle_i < 360:
        margin_arr[0] = load_ratio_arr[0] - ref_left_ratio[angle_idx]
        margin_arr[1] = ref_right_ratio[angle_idx] - load_ratio_arr[1]

    if 0 <= swing_angle_i < 90 or 270 <= swing_angle_i < 360:
        margin_arr[2] = ref_front_ratio[angle_idx] - load_ratio_arr[2]
        margin_arr[3] = load_ratio_arr[3] - ref_rear_ratio[angle_idx]

    if 90 <= swing_angle_i < 270:
        margin_arr[2] = load_ratio_arr[2] - ref_front_ratio[angle_idx]
        margin_arr[3] = ref_rear_ratio[angle_idx] - load_ratio_arr[3]

    return margin_arr


left_margin_list = []
right_margin_list = []
front_margin_list = []
rear_margin_list = []
detection_result_list = []

for i in range(data_sample.shape[0]):
    sample = data_sample.iloc[i, :]
    load_arr = sample[left_load_name_list +right_load_name_list].to_numpy()
    swing_angle = sample['Swing_Angle(deg)']

    margin_arr_output = calculate_margin(ref_left_load_ratio_2, ref_right_load_ratio_2, ref_front_load_ratio, ref_rear_load_ratio, load_arr, swing_angle)

    left_margin_list.append(margin_arr_output[0])
    right_margin_list.append(margin_arr_output[1])
    front_margin_list.append(margin_arr_output[2])
    rear_margin_list.append(margin_arr_output[3])

    detection_result_list.append(int(np.any(margin_arr_output <= -0.05)))
