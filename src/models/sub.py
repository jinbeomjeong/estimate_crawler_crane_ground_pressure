import numpy as np


def calc_roll_over_state(load_cell_arr: np.ndarray) -> tuple[float, int]:
    mean_front_load = np.mean([load_cell_arr[0], load_cell_arr[3]])
    mean_rear_load = np.mean([load_cell_arr[2], load_cell_arr[5]])
    mean_total_front_rear_load = mean_front_load + mean_rear_load

    front_load_ratio = mean_front_load / (mean_total_front_rear_load + 0.001)
    rear_load_ratio = mean_rear_load / (mean_total_front_rear_load + 0.001)
    forward_diff_ratio = np.abs(front_load_ratio - rear_load_ratio)
    roll_over_state = forward_diff_ratio < 0.65
    roll_over_state = roll_over_state.astype(int)

    return forward_diff_ratio, roll_over_state