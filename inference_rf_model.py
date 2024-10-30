import joblib, struct
import numpy as np

from utils.udp_lib import UdpServer


est_model = joblib.load('rf_model.joblib')
udp_handle = UdpServer(address='localhost', port=6341)

while True:
    output_msg = bytes(0)
    read_msg = udp_handle.receive_msg()

    boom_angle = struct.unpack('f', read_msg[0:4])[0]
    swing_angle = struct.unpack('f', read_msg[4:8])[0]
    load_weight = struct.unpack('f', read_msg[8:12])[0]
    roll_angle = struct.unpack('f', read_msg[12:16])[0]
    pitch_angle = struct.unpack('f', read_msg[16:20])[0]

    #  boom+roll/load, boom+pitch/load, 'wing/load, roll/pitch,  load/roll, load/pitch, boom+roll, boom+pitch, boom_x_pos, boom_y_pos
    input_data = np.array([boom_angle, swing_angle, load_weight, roll_angle, pitch_angle,
                           (boom_angle+roll_angle)/load_weight, (boom_angle+pitch_angle)/load_weight, swing_angle/load_weight,
                           roll_angle/pitch_angle, load_weight/roll_angle, load_weight/pitch_angle, boom_angle+roll_angle,
                           boom_angle+pitch_angle, np.cos(np.deg2rad(swing_angle)) * np.cos(np.deg2rad(boom_angle)),
                           np.sin(np.deg2rad(swing_angle)) * np.cos(np.deg2rad(boom_angle))])

    pred_output_arr = est_model.predict(input_data.reshape(1, -1))[0]

    for i, pred_output in enumerate(pred_output_arr):
        output_msg += struct.pack('f', pred_output)

    udp_handle.send_msg(output_msg)

