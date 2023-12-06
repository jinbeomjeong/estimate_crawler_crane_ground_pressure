import time
import numpy as np
from utils.tcp_lib import TCPClient
from xgboost import XGBRegressor


filename = 'xgb_model.json'
model = XGBRegressor()
model.load_model(filename)

tcp_handle = TCPClient(address='localhost', port=6340)
tcp_handle.connect_to_server()

while True:
    input_data = np.fromstring(tcp_handle.receive_msg(), dtype=np.float64, sep=',')
    input_data = input_data.reshape((1, -1))
    pred_output = model.predict(input_data)
    tcp_handle.send_msg(np.array2string(pred_output.flatten('C'), precision=4, separator=',').lstrip('[').rstrip(']'))
    time.sleep(0.2)

