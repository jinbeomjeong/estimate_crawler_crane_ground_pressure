import os, time, can, cantools, canopen, struct
import numpy as np


class LoadCellCANMsgParser:
    def __init__(self, dbc_file_path: str):
        self.__msg_name_list = ['load_cell_2', 'load_cell_3']
        self.__load_cell_arr = np.zeros(shape=(6, ), dtype=np.float32)  # unit: kg
        self.__can_db = cantools.database.load_file(dbc_file_path)

    def get_values(self, packet: can.Message) -> None:
        for msg_name in self.__msg_name_list:
            msg = self.__can_db.get_message_by_name(msg_name)

            if msg.name == self.__msg_name_list[0]:
                if packet.arbitration_id == msg.frame_id:
                    decoded_msg = self.__can_db.decode_message(packet.arbitration_id, packet.data)
                    self.__load_cell_arr[0] = decoded_msg['LC_1']
                    self.__load_cell_arr[1] = decoded_msg['LC_2']
                    self.__load_cell_arr[2] = decoded_msg['LC_3']
                    self.__load_cell_arr[3] = decoded_msg['LC_4']

            if msg.name == self.__msg_name_list[1]:
                if packet.arbitration_id == msg.frame_id:
                    decoded_msg = self.__can_db.decode_message(packet.arbitration_id, packet.data)
                    self.__load_cell_arr[4] = decoded_msg['LC_5']
                    self.__load_cell_arr[5] = decoded_msg['LC_6']

    def read_values(self) -> np.ndarray:
        return self.__load_cell_arr



def setup_can_interface():
    print("CAN interface setting...")
    os.system('sudo ip link set canb1 down')
    os.system('sudo ip link set canb1 type can bitrate 250000')
    os.system('sudo ip link set canb1 up')
    time.sleep(1) # 설정 적용 대기
    print("CAN interface setup complete.")

network = canopen.Network()
setup_can_interface()
network.connect(bustype='socketcan', channel='canb1', bitrate=2500000)

NODE_ID = 10
node = network.add_node(NODE_ID)

while True:
    data_bytes = node.sdo.upload(0x6010, 0)
    raw_val_x = struct.unpack('<h', data_bytes)[0]
    raw_val_x /= 100

    data_bytes = node.sdo.upload(0x6020, 0)
    raw_val_y = struct.unpack('<h', data_bytes)[0]
    raw_val_y /= 100

    print(f"[SDO requst]: {raw_val_x}")
    print(f"[SDO requst]: {raw_val_y}")

    time.sleep(0.1)