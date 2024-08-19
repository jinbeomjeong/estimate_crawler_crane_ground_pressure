import time, struct
import numpy as np
from utils.udp_lib import UdpServer
from xgboost import XGBRegressor


filename = 'xgb_model_v2.json'
model = XGBRegressor()
model.load_model(filename)

udp_handle = UdpServer(address='localhost', port=6341)

msg = bytearray()

while True:
    read_msg = udp_handle.receive_msg()
    boom_angle = struct.unpack('f', read_msg[0:4])[0]
    swing_angle = struct.unpack('f', read_msg[4:8])[0]
    load = struct.unpack('f', read_msg[8:12])[0]

    input_data = np.array([boom_angle, swing_angle, load])
    input_data = input_data.reshape((1, -1))
    pred_output_arr = model.predict(input_data)[0]

    for i, pred_output in enumerate(pred_output_arr):
        msg[(i*4):(i*4)+4] = struct.pack('f', pred_output)

    udp_handle.send_msg(bytes(msg), address='localhost', port=6340)
    time.sleep(0.05)
