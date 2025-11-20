import numpy as np
import pandas as pd

from tqdm.auto import tqdm


margin_arr = np.zeros(4)
load_ratio_arr = np.zeros(4)
ref_angle_arr = np.linspace(start=0, stop=360, num=36001)

left_load_idx_list = [0, 1, 2, 3, 4]
right_load_idx_list = [5, 6, 7, 8, 9]

front_load_idx_list = [0, 1, 5, 6]
rear_load_idx_list = [3, 4, 8, 9]

ref_left_load_ratio = 0.5*np.sin(np.deg2rad(ref_angle_arr))+0.5
ref_left_load_ratio = (0.6*ref_left_load_ratio)+0.2

ref_right_load_ratio = 1 - ref_left_load_ratio

ref_front_load_ratio = 0.3 * np.sin(np.radians(ref_angle_arr) + np.pi / 2) + 0.5
ref_rear_load_ratio = 1 - ref_front_load_ratio


def calculate_load_margin_v1(ground_load:np.array, swing_angle: float):
    global margin_arr, load_ratio_arr

    load_total = np.sum(ground_load)
    left_load_total = np.sum(ground_load[left_load_idx_list])
    right_load_total = np.sum(ground_load[right_load_idx_list])

    front_view_load_total = np.sum(ground_load[front_load_idx_list + rear_load_idx_list])
    front_load_total = np.sum(ground_load[front_load_idx_list])
    rear_load_total = np.sum(ground_load[rear_load_idx_list])

    load_ratio_arr[0] = (left_load_total / load_total)
    load_ratio_arr[1] = (right_load_total / load_total)
    load_ratio_arr[2] = (front_load_total / load_total)
    load_ratio_arr[3] = (rear_load_total / load_total)

    angle_idx = np.argmin(np.abs(ref_angle_arr - swing_angle))

    if 0 <= swing_angle < 180:
        margin_arr[0] = ref_left_load_ratio[angle_idx] - load_ratio_arr[0]
        margin_arr[1] = load_ratio_arr[1] - ref_right_load_ratio[angle_idx]

    if 180 <= swing_angle < 360:
        margin_arr[0] = load_ratio_arr[0] - ref_left_load_ratio[angle_idx]
        margin_arr[1] = ref_right_load_ratio[angle_idx] - load_ratio_arr[1]

    if 0 <= swing_angle < 90 or 270 <= swing_angle < 360:
        margin_arr[2] = ref_front_load_ratio[angle_idx] - load_ratio_arr[2]
        margin_arr[3] = load_ratio_arr[3] - ref_rear_load_ratio[angle_idx]

    if 90 <= swing_angle < 270:
        margin_arr[2] = load_ratio_arr[2] - ref_front_load_ratio[angle_idx]
        margin_arr[3] = ref_rear_load_ratio[angle_idx] - load_ratio_arr[3]

    roll_over_det = np.any(margin_arr <= -0.05)

    return margin_arr, roll_over_det


def calculate_load_margin_v2(dataset: pd.DataFrame):
    ref_swing_angle_list = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
    target_load_ratio_list = np.array([0.3, 0.45, 0.2, 0.45, 0.3, 0.45, 0.2, 0.45, 0.3])

    left_load_name_list = ['Actual_Load_Left_1(N)', 'Actual_Load_Left_2(N)', 'Actual_Load_Left_3(N)',
                           'Actual_Load_Left_4(N)', 'Actual_Load_Left_5(N)']
    right_load_name_list = ['Actual_Load_Right_1(N)', 'Actual_Load_Right_2(N)', 'Actual_Load_Right_3(N)',
                            'Actual_Load_Right_4(N)', 'Actual_Load_Right_5(N)']

    left_load_idx_list = [0, 1, 2, 3, 4]
    right_load_idx_list = [5, 6, 7, 8, 9]

    front_load_idx_list = [0, 1, 5, 6]
    rear_load_idx_list = [3, 4, 8, 9]

    mean_left_load_list = []
    mean_right_load_list = []
    mean_front_load_list = []
    mean_rear_load_list = []
    roll_over_det_list = []
    cal_det_ratio_list = []

    n_file = int(dataset['file_idx'].max() + 1)

    for file_idx in tqdm(range(n_file)):
        data_sample = dataset[dataset['file_idx'] == file_idx]

        for i in range(data_sample.shape[0]):
            sample = data_sample.iloc[i, :]
            load_arr = sample[left_load_name_list + right_load_name_list].to_numpy()
            swing_angle = sample['Swing_Angle(deg)'].item()
            target_load_ratio = np.interp(swing_angle, ref_swing_angle_list, target_load_ratio_list)

            mean_left_load = np.mean(load_arr[left_load_idx_list])
            mean_right_load = np.mean(load_arr[right_load_idx_list])
            mean_front_load = np.mean(load_arr[front_load_idx_list])
            mean_rear_load = np.mean(load_arr[rear_load_idx_list])
            mean_total_load = np.mean(load_arr)
            mean_front_rear_load = np.mean(load_arr[front_load_idx_list + rear_load_idx_list])

            left_load_margin = mean_left_load / mean_total_load
            right_load_margin = mean_right_load / mean_total_load
            front_load_margin = mean_front_load / mean_front_rear_load
            rear_load_margin = mean_rear_load / mean_front_rear_load

            roll_over_det = (left_load_margin < target_load_ratio) | (right_load_margin < target_load_ratio) | (
                        front_load_margin < target_load_ratio) | (rear_load_margin < target_load_ratio)
            roll_over_det = roll_over_det.astype(int)

            mean_left_load_list.append(left_load_margin)
            mean_right_load_list.append(right_load_margin)
            mean_front_load_list.append(front_load_margin)
            mean_rear_load_list.append(rear_load_margin)
            roll_over_det_list.append(roll_over_det)
            cal_det_ratio_list.append(target_load_ratio)

    mean_left_load_arr = np.array(mean_left_load_list)
    mean_right_load_arr = np.array(mean_right_load_list)
    mean_front_load_arr = np.array(mean_front_load_list)
    mean_rear_load_arr = np.array(mean_rear_load_list)
    roll_over_det_arr = np.array(roll_over_det_list)

    result_df = pd.DataFrame(np.concatenate([mean_left_load_arr.reshape(-1, 1), mean_right_load_arr.reshape(-1, 1),
                                             mean_front_load_arr.reshape(-1, 1),mean_rear_load_arr.reshape(-1, 1),
                                             roll_over_det_arr.reshape(-1, 1)], axis=1),
                             columns=['mean_left_load', 'mean_right_load', 'mean_front_load', 'mean_rear_load','roll_over_det'])

    return result_df
