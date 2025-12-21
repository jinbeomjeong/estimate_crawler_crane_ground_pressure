import logging, time, struct
import onnxruntime as ort
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

from src.models.sub import calc_roll_over_state
from src.models.parameter import topic_dict
from tqdm.auto import tqdm

angle_data = {'x_axis': 0.0, 'y_axis': 0.0}
seq_len = 50
pred_distance = 0
roll_over_state = 0
detection = 0


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = ort.InferenceSession('outputs/checkpoints/model_seq_50_pred_0.onnx')
logger.info('onnx model loaded!')

# BROKER_ADDRESS = '192.168.0.2'
# client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# client.connect(host=BROKER_ADDRESS, port=1883)
# client.loop_start()
# logger.info('mqtt message publish client started!')

logging_data = pd.DataFrame()
data_name_list = ['time(sec)', 'boom_length(m)', 'boom_angle(deg)', 'load_weight(ton)', 'engine_speed(rpm)',
                  'wind_speed(m/s)', 'swing_angle(deg)', 'body_angle_x(deg)', 'body_angle_y(deg)',
                  'load_cell_left_1', 'load_cell_left_2', 'load_cell_left_3', 'load_cell_right_1', 'load_cell_right_2', 'load_cell_right_3',
                  'detection']

load_cell_name_list = ['load_cell_left_1', 'load_cell_left_2', 'load_cell_left_3',
                       'load_cell_right_1', 'load_cell_right_2', 'load_cell_right_3']

feature_name_list = ['boom_angle(deg)', 'load_weight(ton)', 'engine_speed(rpm)']
target_name = 'roll_over_state'
input_buf = np.zeros(shape=(1, seq_len, len(feature_name_list)), dtype=np.float32)

sample_data = pd.read_csv('data/on-road_test/data20251125-095900.csv')
t0 = time.perf_counter()

for i in range(sample_data.shape[0]):
    prv_time = time.perf_counter()
    relative_time = prv_time - t0

    input_sig = sample_data.iloc[i, :][feature_name_list].to_numpy()
    boom_angle = input_sig[0]
    load_weight = input_sig[1]
    engine_speed = input_sig[2]

    load_cell_arr = sample_data.iloc[i, :][load_cell_name_list].to_numpy()
    load_ratio, roll_over_state = calc_roll_over_state(load_cell_arr=load_cell_arr)

    input_buf = np.roll(a=input_buf, shift=-1, axis=1)
    input_buf[0, -1, :] = input_sig

    pred = np.square(model.run(output_names=None, input_feed={'input': input_buf})).item()
    detection = int(pred > 0.01)

    print(f"{relative_time:.2f}", boom_angle, load_weight, detection, roll_over_state)
    print(end='\n\n')

    # client.publish(topic=topic_dict['time_topic'], payload=struct.pack('<f', relative_time))
    # client.publish(topic=topic_dict['boom_angle_topic'], payload=struct.pack('<f', boom_angle))
    # client.publish(topic=topic_dict['load_weight_topic'], payload=struct.pack('<f', load_weight))
    # client.publish(topic=topic_dict['engine_speed_topic'], payload=struct.pack('<f', engine_speed))
    # client.publish(topic=topic_dict['left_load_1_topic'], payload=struct.pack('<f', load_cell_arr[0].item()))
    # client.publish(topic=topic_dict['left_load_2_topic'], payload=struct.pack('<f', load_cell_arr[1].item()))
    # client.publish(topic=topic_dict['left_load_3_topic'], payload=struct.pack('<f', load_cell_arr[2].item()))
    # client.publish(topic=topic_dict['right_load_1_topic'], payload=struct.pack('<f', load_cell_arr[3].item()))
    # client.publish(topic=topic_dict['right_load_2_topic'], payload=struct.pack('<f', load_cell_arr[4].item()))
    # client.publish(topic=topic_dict['right_load_3_topic'], payload=struct.pack('<f', load_cell_arr[5].item()))
    # client.publish(topic=topic_dict['detection_topic'], payload=struct.pack('<f', detection))
    # client.publish(topic=topic_dict['front_load_ratio_topic'], payload=struct.pack('<f', load_ratio))
    # client.publish(topic=topic_dict['roll_over_topic'], payload=struct.pack('<f', roll_over_state))

    time.sleep(0.05)

