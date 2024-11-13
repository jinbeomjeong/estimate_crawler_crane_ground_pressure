import numpy as np


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


def calculate_load_margin(ground_load:np.array, swing_angle: float):
    global margin_arr, load_ratio_arr

    load_total = np.sum(ground_load)
    left_load_total = np.sum(ground_load[left_load_idx_list])
    right_load_total = np.sum(ground_load[right_load_idx_list])

    front_view_load_total = np.sum(ground_load[front_load_idx_list + rear_load_idx_list])
    front_load_total = np.sum(ground_load[front_load_idx_list])
    rear_load_total = np.sum(ground_load[rear_load_idx_list])

    load_ratio_arr[0] = (left_load_total / load_total)
    load_ratio_arr[1] = (right_load_total / load_total)
    load_ratio_arr[2] = (front_load_total / front_view_load_total)
    load_ratio_arr[3] = (rear_load_total / front_view_load_total)

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
