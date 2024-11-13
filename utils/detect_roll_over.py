import numpy as np

left_load_idx_list = [0, 1, 2, 3, 4]
right_load_idx_list = [5, 6, 7, 8, 9]

front_load_idx_list = [0, 1, 5, 6]
rear_load_idx_list = [3, 4, 8, 9]

angle_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
left_ref_load_list = [50, 70, 80, 70, 50, 30, 20, 30, 50]
right_ref_load_list = [50, 30, 20, 30, 50, 70, 80, 70, 50]
front_ref_load_list = [60, 90, 100, 90, 60, 90, 100, 90, 60]


def calculate_side_view_load_margin(ref_load_ratio, actual_load_ratio):
    if ref_load_ratio >= 50:
        margin = ref_load_ratio - actual_load_ratio
    else:
        margin = actual_load_ratio - ref_load_ratio

    return 1-(margin / ref_load_ratio)


def calculate_front_view_margin(ref_load_ratio, actual_load_ratio):
    return 1-((ref_load_ratio - actual_load_ratio) / ref_load_ratio)


def detect_roll_over(ground_load: np.array, swing_angle:float):
    side_view_load_total = np.sum(ground_load)
    left_load_total = np.sum(ground_load[left_load_idx_list])
    right_load_total = np.sum(ground_load[right_load_idx_list])

    front_view_load_total = np.sum(ground_load[front_load_idx_list+rear_load_idx_list])
    front_load_total = np.sum(ground_load[front_load_idx_list])
    rear_load_total = np.sum(ground_load[rear_load_idx_list])

    left_load_ratio = (left_load_total / side_view_load_total)*100
    right_load_ratio = (right_load_total / side_view_load_total)*100
    front_load_ratio = (front_load_total / front_view_load_total)*100
    rear_load_ratio = (rear_load_total / front_view_load_total)*100

    ref_left_load_ratio = np.interp(swing_angle, angle_list, left_ref_load_list)
    ref_right_load_ratio = np.interp(swing_angle, angle_list, right_ref_load_list)
    ref_front_load_ratio = np.interp(swing_angle, angle_list, front_ref_load_list)

    left_load_margin = calculate_side_view_load_margin(ref_left_load_ratio, left_load_ratio)
    right_load_margin = calculate_side_view_load_margin(ref_right_load_ratio, right_load_ratio)
    front_load_margin = calculate_front_view_margin(ref_front_load_ratio, front_load_ratio)
    rear_load_margin = calculate_front_view_margin(ref_front_load_ratio, rear_load_ratio)

    return left_load_margin, right_load_margin, front_load_margin, rear_load_margin


