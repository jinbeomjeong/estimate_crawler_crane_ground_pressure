import logging, threading, time, struct, can, canopen, datetime, socket
import onnxruntime as ort
import numpy as np
import pandas as pd

from src.models.parameter import topic_dict
from pymodbus.server import StartSerialServer
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusDeviceContext, ModbusServerContext
from src.msg_parser import LoadCellCANMsgParser, byte_list_parser
from src.miscellaneous import setup_can_interface


angle_data = {'x_axis': 0.0, 'y_axis': 0.0}
seq_len = 50
pred_distance = 0
load_cell_arr_fix = np.zeros(shape=(6, ), dtype=np.float32)
input_buf = np.zeros(shape=(1, seq_len, 6), dtype=np.float32)

pred = 0.0
detection = 0
ref_swing_angle = 0.0
load_ratio = 0.0
roll_over_state = 0
udp_data = np.zeros(shape=(26, ), dtype=np.float32)


def modbus_com():
    logger.info("modbus server started!")
    StartSerialServer(context=context, port='/dev/com2', baudrate=115200,  bytesize=8, parity='N', stopbits=1)

def can_com_1():
    for can_msg in can_ch_1:
        load_cell.get_values(packet=can_msg)

def udp_sender():
    server_ip = "192.168.10.100"
    server_port = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    while True:
        try:
            msg = struct.pack('<26f', *udp_data)
            sock.sendto(msg, (server_ip, server_port))
            time.sleep(0.1)

        except socket.timeout:
            print("No Server Response. reconnecting...")
            time.sleep(1)

        except Exception as e:
            print(f'udp system error: {e}')
            time.sleep(1)


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = ort.InferenceSession('outputs/checkpoints/model_seq_50_pred_0_15m.onnx')
logger.info('onnx model loaded!')

pymodbus_log = logging.getLogger("pymodbus")
pymodbus_log.setLevel(logging.CRITICAL)

store = ModbusSequentialDataBlock(address=0, values=[0] * 100)
slave_context = ModbusDeviceContext(hr=store)
context = ModbusServerContext(devices={1: slave_context}, single=False)

setup_can_interface(channel='canb0', bitrate=250000)
setup_can_interface(channel='canb1', bitrate=250000)

can_ch_1 = can.interface.Bus(interface='socketcan', channel='canb0', bitrate=250000)
load_cell = LoadCellCANMsgParser('src/utils/load_cell.dbc')

modbus_com_task = threading.Thread(target=modbus_com)
modbus_com_task.daemon = True
modbus_com_task.start()

can_ch1_com_task = threading.Thread(target=can_com_1)
can_ch1_com_task.daemon = True
can_ch1_com_task.start()
logger.info('can channel 1 communication started!')

network = canopen.Network()
network.connect(bustype='socketcan', channel='canb1', bitrate=250000)
node = network.add_node(10)
logger.info('can channel 2 communication started!')

udp_com_task = threading.Thread(target=udp_sender)
udp_com_task.daemon = True
udp_com_task.start()
logger.info('udp communication started!')

# BROKER_ADDRESS = '192.168.137.172'
# client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# client.connect(host=BROKER_ADDRESS, port=1883)
# client.loop_start()
# logger.info('mqtt message publish client started!')

logging_data = pd.DataFrame()
data_name_list = ['time(sec)', 'boom_length(m)', 'boom_angle(deg)', 'load_weight(ton)', 'engine_speed(rpm)',
                  'wind_speed(m/s)', 'swing_angle(deg)', 'body_angle_x(deg)', 'body_angle_y(deg)',
                  'load_cell_left_1', 'load_cell_left_2', 'load_cell_left_3', 'load_cell_right_1', 'load_cell_right_2', 'load_cell_right_3',
                  'load_ratio', 'roll_over_state', 'pred', 'detection']

log_file_name = 'log_data/data' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
log_data_header = pd.DataFrame(columns=data_name_list)
log_data_header.to_csv(log_file_name, mode='a', header=True)

logger.info("program started!")

t0 = time.perf_counter()

while True:
    prv_time = time.perf_counter()
    relative_time = prv_time - t0

    register_values = store.getValues(address=0, count=50)

    boom_length = byte_list_parser([register_values[3], register_values[2]])
    udp_data[7] = boom_length

    boom_angle = byte_list_parser([register_values[5], register_values[4]])
    udp_data[8] = boom_angle

    specifications = byte_list_parser([register_values[7], register_values[6]])
    udp_data[9] = specifications

    Radius_MAIN = byte_list_parser([register_values[9], register_values[8]])
    udp_data[10] = Radius_MAIN

    Radius_AUX = byte_list_parser([register_values[11], register_values[10]])
    udp_data[11] = Radius_AUX

    load_weight = byte_list_parser([register_values[13], register_values[12]])
    udp_data[12] = load_weight

    battery_voltage = byte_list_parser([register_values[15], register_values[14]])
    udp_data[13] = battery_voltage

    engine_speed = byte_list_parser([register_values[17], register_values[16]])
    udp_data[14] = engine_speed

    engine_temp = byte_list_parser([register_values[19], register_values[18]])
    udp_data[15] = engine_temp

    oil_pressure = byte_list_parser([register_values[21], register_values[20]])
    udp_data[16] = oil_pressure

    Working_oil_temp = byte_list_parser([register_values[23], register_values[22]])
    udp_data[17] = Working_oil_temp

    main_height = byte_list_parser([register_values[25], register_values[24]])
    udp_data[18] = main_height

    aux_height = byte_list_parser([register_values[27], register_values[26]])
    udp_data[19] = aux_height

    _rd_height = byte_list_parser([register_values[29], register_values[28]])
    udp_data[20] = _rd_height

    status_1 = byte_list_parser([register_values[31], register_values[30]])
    udp_data[21] = status_1

    status_2 = byte_list_parser([register_values[33], register_values[32]])
    udp_data[22] = status_2

    lower_angle = byte_list_parser([register_values[37], register_values[36]])
    udp_data[23] = lower_angle

    wind_speed = byte_list_parser([register_values[35], register_values[34]])
    udp_data[24] = wind_speed

    swing_angle = byte_list_parser([register_values[39], register_values[38]])
    udp_data[25] = swing_angle

    data_bytes = node.sdo.upload(0x6010, 0)
    raw_val_x = struct.unpack('<h', data_bytes)[0]
    body_x_axis_angle = raw_val_x/100

    data_bytes = node.sdo.upload(0x6020, 0)
    raw_val_y = struct.unpack('<h', data_bytes)[0]
    body_y_axis_angle = raw_val_y/100

    load_cell_arr = load_cell.read_values()
    load_cell_arr_fix[0] = load_cell_arr[0]
    load_cell_arr_fix[1] = load_cell_arr[4]
    load_cell_arr_fix[2] = load_cell_arr[3]
    load_cell_arr_fix[3] = load_cell_arr[1]
    load_cell_arr_fix[4] = load_cell_arr[5]
    load_cell_arr_fix[5] = load_cell_arr[2]
    
    for i in range(len(load_cell_arr_fix)):
        udp_data[i] = load_cell_arr_fix[i].item()

    input_buf = np.roll(a=input_buf, shift=-1, axis=1)
    input_buf[0, -1, :] = np.array([boom_angle, ref_swing_angle, load_weight, engine_speed,
                                    body_x_axis_angle, body_y_axis_angle], dtype=np.float32)

    if engine_speed > 300:
        pred = np.squeeze(model.run(output_names=None, input_feed={'input': input_buf})).item()
        detection = (pred > 0.91)*1
    else:
        detection = 0
        pred = 0

    udp_data[6] = detection

    if True:
        left_sum = load_cell_arr_fix[0] + load_cell_arr_fix[2]
        left_1 = load_cell_arr_fix[0] / (left_sum+0.001)
        left_2 = load_cell_arr_fix[2] / (left_sum+0.001)
        left = left_1 - left_2

        right_sum = load_cell_arr_fix[3] + load_cell_arr_fix[5]
        right_1 = load_cell_arr_fix[3] / (right_sum+0.001)
        right_2 = load_cell_arr_fix[5] / (right_sum+0.001)
        right = right_1 - right_2

        load_ratio = left * right
        roll_over_state = (load_ratio < -0.65) * 1

    else:
        left_sum = load_cell_arr_fix[3] + load_cell_arr_fix[0]
        left_1 = load_cell_arr_fix[3] / (left_sum+0.001)
        left_2 = load_cell_arr_fix[0] / (left_sum+0.001)
        left = left_1 - left_2

        right_sum = load_cell_arr_fix[5] + load_cell_arr_fix[2]
        right_1 = load_cell_arr_fix[5] / (right_sum+0.001)
        right_2 = load_cell_arr_fix[2] / (right_sum+0.001)
        right = right_1 - right_2

        load_ratio = left * right
        roll_over_state = (load_ratio < 0.2) * 1

    logging_data = pd.DataFrame(data={data_name_list[0]: round(relative_time, 3),
                                      data_name_list[1]: round(boom_length, 3),
                                      data_name_list[2]: round(boom_angle, 3),
                                      data_name_list[3]: round(load_weight, 3),
                                      data_name_list[4]: round(engine_speed, 3),
                                      data_name_list[5]: round(wind_speed, 3),
                                      data_name_list[6]: round(swing_angle, 3),
                                      data_name_list[7]: round(body_x_axis_angle, 3),
                                      data_name_list[8]: round(body_y_axis_angle, 3),
                                      data_name_list[9]: round(load_cell_arr_fix[0].item(), 3),
                                      data_name_list[10]: round(load_cell_arr_fix[1].item(), 3),
                                      data_name_list[11]: round(load_cell_arr_fix[2].item(), 3),
                                      data_name_list[12]: round(load_cell_arr_fix[3].item(), 3),
                                      data_name_list[13]: round(load_cell_arr_fix[4].item(), 3),
                                      data_name_list[14]: round(load_cell_arr_fix[5].item(), 3),
                                      data_name_list[15]: round(load_ratio, 5),
                                      data_name_list[16]: roll_over_state,
                                      data_name_list[17]: round(pred,5),
                                      data_name_list[18]: detection}, index=[0])
    logging_data.to_csv(log_file_name, mode='a', header=False)

    # client.publish(topic=topic_dict['time_topic'], payload=struct.pack('<f', relative_time))
    # client.publish(topic=topic_dict['boom_angle_topic'], payload=struct.pack('<f', boom_angle))
    # client.publish(topic=topic_dict['load_weight_topic'], payload=struct.pack('<f', load_weight))
    # client.publish(topic=topic_dict['swing_angle_topic'], payload=struct.pack('<f', swing_angle))
    # client.publish(topic=topic_dict['engine_speed_topic'], payload=struct.pack('<f', engine_speed))
    # client.publish(topic=topic_dict['body_angle_x_topic'], payload=struct.pack('<f', angle_data['x_axis']))
    # client.publish(topic=topic_dict['body_angle_y_topic'], payload=struct.pack('<f', angle_data['y_axis']))
    # client.publish(topic=topic_dict['left_load_1_topic'], payload=struct.pack('<f', load_cell_arr_fix[0].item()))
    # client.publish(topic=topic_dict['left_load_2_topic'], payload=struct.pack('<f', load_cell_arr_fix[1].item()))
    # client.publish(topic=topic_dict['left_load_3_topic'], payload=struct.pack('<f', load_cell_arr_fix[2].item()))
    # client.publish(topic=topic_dict['right_load_1_topic'], payload=struct.pack('<f', load_cell_arr_fix[3].item()))
    # client.publish(topic=topic_dict['right_load_2_topic'], payload=struct.pack('<f', load_cell_arr_fix[4].item()))
    # client.publish(topic=topic_dict['right_load_3_topic'], payload=struct.pack('<f', load_cell_arr_fix[5].item()))
    # client.publish(topic=topic_dict['detection_topic'], payload=struct.pack('<f', detection))
    # client.publish(topic=topic_dict['front_load_ratio_topic'], payload=struct.pack('<f', load_ratio))
    # client.publish(topic=topic_dict['roll_over_topic'], payload=struct.pack('<f', roll_over_state))

    period_time = time.perf_counter() - prv_time

    if period_time >= 0.1:
        delay_time = 0
    else:
        delay_time = 0.1 - period_time

    time.sleep(delay_time)

    logger.info(f'relative time: {relative_time:.2f}sec, period time: {period_time*1000:.2f}msec')
    # print(f'{boom_angle:.2f}', f'{load_weight:.2f}', f'{body_x_axis_angle:.2f}', f'{body_y_axis_angle:.2f}',
    #       f'{load_cell_arr_fix}', f'{load_ratio:.2f}', roll_over_state, f'{pred:.3f}', detection)