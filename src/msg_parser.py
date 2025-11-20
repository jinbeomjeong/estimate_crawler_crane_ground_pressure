import struct
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