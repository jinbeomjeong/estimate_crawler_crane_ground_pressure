import os, re#, cv2, pytesseract
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import RobustScaler


val_data_file_name_list = ['safe-boom-65-swing-180-load-80.csv', 'safe-boom-75-swing-180-load-110'
                           'unsafe-swing-0-load-70.csv', 'unsafe-swing-45-load-70.csv', 'unsafe-swing-90-load-70.csv',
                           'unsafe-swing-135-load-70.csv', 'unsafe-swing-180-load-70.csv']

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# feature_names = ['swing_angle(deg)', 'boom_angle(deg)', 'weight(ton)']
# target_names = ['fl_pressure(kg/cm^2)', 'fr_pressure(kg/cm^2)', 'rl_pressure(kg/cm^2)', 'rr_pressure(kg/cm^2)', 'left_pressure_length(m)', 'right_pressure_length(m)']
#
#
# def img_to_dataset(img_path_list: list) -> np.array:
#     dataset_length = len(img_path_list)
#     target_status = np.zeros(shape=(dataset_length, len(target_names)), dtype=np.float64)
#     feature_status = np.zeros(shape=(dataset_length, len(feature_names)), dtype=np.float64)
#
#     for i, img_path in enumerate(tqdm(img_path_list, ncols=100, desc='data analyzing...')):
#         feature_parameter = img_path[img_path.rfind(os.sep) + 1:].rstrip('.jpg')
#
#         swing_angle = int(feature_parameter.split('swing-')[1].split('-boom')[0])
#         boom_angle = int(feature_parameter.split('boom-')[1].split('-weight')[0])
#         weight = int(feature_parameter.split('weight-')[1].split('-')[0])
#
#         feature_status[i, 0] = swing_angle
#         feature_status[i, 1] = boom_angle
#         feature_status[i, 2] = weight
#
#         img = cv2.imread(img_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         fl_p_img = gray[628:646, 2220:2305]
#         fr_p_img = gray[628:646, 2462:2545]
#         rl_p_img = gray[814:830, 2220:2305]
#         rr_p_img = gray[814:830, 2462:2545]
#
#         pressure_imgz = [fl_p_img, fr_p_img, rl_p_img, rr_p_img]
#
#         for i, pressure_img in enumerate(pressure_imgz):
#             p_str = pytesseract.image_to_string(pressure_img)
#
#             try:
#                 target_status[i, 0] = float(re.findall('\d+.\d+', p_str)[0])
#
#             except:
#                 target_status[i, 0] = -1
#
#         left_length_img = np.transpose(np.flip(gray[662:808, 2260:2280], axis=0))
#         right_length_img = np.transpose(np.flip(gray[662:808, 2485:2505], axis=0))
#
#         length_imgz = [left_length_img, right_length_img]
#
#         for i, length_img in enumerate(length_imgz):
#             length_str = pytesseract.image_to_string(length_img)
#
#             try:
#                 target_status[i, 1] = float(re.findall('\d+.\d+', length_str)[0])
#
#                 if 6.75 < target_status[i, 1]:
#                     target_status[i, 1] = target_status[i, 1] / 100
#             except:
#                 target_status[i, 1] = -1
#
#     return pd.DataFrame(np.hstack((feature_status, target_status)), columns=feature_names+target_names)


class CraneDataset:
    def __init__(self, file_path_list: list):
        self.__data_file_name_list = file_path_list
        self.__data_feature_names = ['Boom_Angle(deg)', 'Swing_Angle(deg)', 'Load(Ton)', 'Roll_Angle(deg)', 'Yaw_Angle(deg)', 'Pitch_Angle(deg)']
        self.__data_target_names = ['Actual_Load_Left_1(N)', 'Actual_Load_Left_2(N)', 'Actual_Load_Left_3(N)',
                                    'Actual_Load_Left_4(N)', 'Actual_Load_Left_5(N)', 'Actual_Load_Right_1(N)',
                                    'Actual_Load_Right_2(N)', 'Actual_Load_Right_3(N)', 'Actual_Load_Right_4(N)',
                                    'Actual_Load_Right_5(N)']
        self.__extra_col_names = ['file_idx', 'safe_state', 'dataset_type']
        self.__model_feature_names = ['f10', 'f11', 'f12']
        self.__model_target_names = ['f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29']

        self.__raw_dataset = self.load_dataset(self.__data_file_name_list)

    @staticmethod
    def get_load_value(data_file_name: str) -> int:
        match = re.search(r'load-(\d+)', data_file_name)

        return int(match.group(1))

    @staticmethod
    def get_safe_state(data_file_name: str) -> int:
        safe_state = 0
        idx = data_file_name.find('-')

        if data_file_name[:idx] == 'unsafe':
            safe_state = 1

        return safe_state

    def filtered_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame():
        filtered_dataset = dataset.copy()

        filtered_dataset = filtered_dataset[filtered_dataset['Time(sec)'] > 2.0]

        for target_name in self.__data_target_names:
            filtered_dataset = filtered_dataset[filtered_dataset[target_name] > 1000]

        return filtered_dataset

    def load_dataset(self, data_file_path_list: list) -> pd.DataFrame:
        data_header = ['Time(sec)'] + self.__extra_col_names + self.__data_feature_names + self.__data_target_names
        raw_dataset = []

        for i, file_path in enumerate(tqdm(data_file_path_list)):
            file_name = os.path.basename(file_path)

            if np.any(file_name == np.array(val_data_file_name_list)):
                dataset_type = 1
            else: dataset_type = 0

            file_name = os.path.basename(file_path)
            raw = pd.read_csv(file_path, encoding='ISO-8859-1')
            raw = raw.to_numpy()

            time_arr = raw[:, 0].reshape(-1, 1)
            ground_load_arr = raw[:, 1:11]
            angle_arr = raw[:, 11:13]
            angular_pos_arr = raw[:, 14:]

            load = self.get_load_value(file_name)
            load_arr = np.full(shape=(raw.shape[0], 1), fill_value=load, dtype=np.float32)
            file_idx_arr = np.full(shape=(raw.shape[0], 1), fill_value=i, dtype=np.uint8)
            safe_state = self.get_safe_state(file_name)
            safe_state_arr = np.full(shape=(raw.shape[0], 1), fill_value=safe_state, dtype=np.uint8)
            dataset_type_arr = np.full(shape=(raw.shape[0], 1), fill_value=dataset_type, dtype=np.uint8)

            raw_data = np.concatenate([time_arr, file_idx_arr, safe_state_arr, dataset_type_arr, angle_arr,
                                       load_arr, angular_pos_arr, ground_load_arr], axis=1)
            raw_dataset.append(raw_data)

        raw_dataset = pd.DataFrame(np.vstack(raw_dataset), columns=data_header)
        raw_dataset['Boom_Angle(deg)'] = raw_dataset['Boom_Angle(deg)'] + 70

        return raw_dataset

    def get_dataset(self) -> pd.DataFrame:
        return self.__raw_dataset

    def get_data_feature_names(self) -> list:
        return self.__data_feature_names

    def get_data_target_names(self) -> list:
        return self.__data_target_names

    def get_model_feature_names(self) -> list:
        return self.__model_feature_names

    def get_model_target_names(self) -> list:
        return self.__model_target_names


def create_lstm_dataset(data, seq_len=1, pred_distance=1, target_idx_pos=1):
    feature, target = [], []

    for i in range(data.shape[0] - pred_distance):
        if i + 1 >= seq_len:
            feature.append(data[i + 1 - seq_len:i + 1, :])
            target.append(data[i + pred_distance, target_idx_pos])

    return np.array(feature), np.array(target)


def normalize_train_dataset(dataset: dict, file_name_list: list) -> tuple:
    feature_dataset = dict()
    target_dataset = dict()

    feature_arr_list = []
    target_arr_list = []

    for file_name in file_name_list:
        feature_dataset[file_name] = dataset[file_name][:, 0:15]
        target_dataset[file_name] = dataset[file_name][:, 15:]

    for file_name in file_name_list:
        feature_arr_list.append(feature_dataset[file_name])
        target_arr_list.append(target_dataset[file_name])

    feature_total_data = np.vstack(feature_arr_list)
    target_total_data = np.vstack(target_arr_list).T.flatten().reshape(-1, 1)

    feature_scaler = RobustScaler().fit(feature_total_data)
    target_scaler = RobustScaler().fit(target_total_data)

    scaled_dataset = dict()

    scaled_feature = []
    scaled_target = []

    for file_name in file_name_list:
        temp = []
        feature_data = feature_dataset[file_name]
        target_data = target_dataset[file_name]

        scaled_feature.append(feature_scaler.transform(feature_data))

        for i in range(target_data.shape[1]):
            out = np.squeeze(target_scaler.transform(target_data[:, i].reshape(-1, 1)))
            temp.append(out)

        scaled_target.append(np.stack(temp, axis=1))

    for i, file_name in enumerate(file_name_list):
        scaled_dataset[file_name] = np.hstack([scaled_feature[i], scaled_target[i]])

    return scaled_dataset, (feature_scaler, target_scaler)


def normalize_val_dataset(dataset: dict, file_name_list: list,
                          feature_scaler: RobustScaler, target_scaler: RobustScaler) -> dict:
    feature_dataset = dict()
    target_dataset = dict()

    for file_name in file_name_list:
        feature_dataset[file_name] = dataset[file_name][:, 0:15]
        target_dataset[file_name] = dataset[file_name][:, 15:]

    scaled_dataset = dict()

    scaled_feature = []
    scaled_target = []

    for file_name in file_name_list:
        temp = []
        feature_data = feature_dataset[file_name]
        target_data = target_dataset[file_name]

        scaled_feature.append(feature_scaler.transform(feature_data))

        for i in range(target_data.shape[1]):
            out = np.squeeze(target_scaler.transform(target_data[:, i].reshape(-1, 1)))
            temp.append(out)

        scaled_target.append(np.stack(temp, axis=1))

    for i, file_name in enumerate(file_name_list):
        scaled_dataset[file_name] = np.hstack([scaled_feature[i], scaled_target[i]])

    return scaled_dataset