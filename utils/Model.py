import joblib
import numpy as np


est_model = joblib.load('d:\\Workspace\\Python\\estimate_crawler_crane_ground_pressure\\rf_model.joblib')

def execute_model(crane_state_arr: np.array):
    boom_angle, swing_angle, load_weight, roll_angle, pitch_angle = (crane_state_arr[0], crane_state_arr[1],
                                                                     crane_state_arr[2], crane_state_arr[3], crane_state_arr[4])

    if load_weight != 0 and roll_angle != 0 and pitch_angle != 0:
        #  boom+roll/load, boom+pitch/load, 'wing/load, roll/pitch,  load/roll, load/pitch, boom+roll, boom+pitch, boom_x_pos, boom_y_pos
        input_data = np.array([boom_angle, swing_angle, load_weight, roll_angle, pitch_angle,
                               (boom_angle +roll_angle ) /load_weight, (boom_angle +pitch_angle ) /load_weight, swing_angle /load_weight,
                               roll_angle /pitch_angle, load_weight /roll_angle, load_weight /pitch_angle, boom_angle +roll_angle,
                               boom_angle +pitch_angle, np.cos(np.deg2rad(swing_angle)) * np.cos(np.deg2rad(boom_angle)),
                               np.sin(np.deg2rad(swing_angle)) * np.cos(np.deg2rad(boom_angle))])
    else:
        input_data = np.array([1 ] *15)

    return est_model.predict(input_data.reshape(1, -1))[0]

