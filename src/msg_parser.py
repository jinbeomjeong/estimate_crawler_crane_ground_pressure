import struct, can, cantools
import numpy as np


def crane_state_msg_parse(msg: bytearray) -> np.array:
    boom_angle = struct.unpack('f', msg[0:4])[0]
    swing_angle = struct.unpack('f', msg[4:8])[0]
    load_weight = struct.unpack('f', msg[8:12])[0]
    roll_angle = struct.unpack('f', msg[12:16])[0]
    pitch_angle = struct.unpack('f', msg[16:20])[0]

    return np.array([boom_angle, swing_angle, load_weight, roll_angle, pitch_angle])


def crane_under_load_parse(load: np.array) -> bytearray:
    output_msg = bytes(0)

    for i, pred_output in enumerate(load):
        output_msg += struct.pack('f', pred_output)

    return output_msg


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
