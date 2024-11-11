import os, re#, cv2, pytesseract
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import RobustScaler


val_data_file_name_list = ['safe-boom-65-swing-180-load-80.csv', 'safe-boom-75-swing-180-load-110'
                           'unsafe-swing-0-load-90.csv', 'unsafe-swing-45-load-70.csv', 'unsafe-swing-90-load-100.csv',
                           'unsafe-swing-135-load-50.csv', 'unsafe-swing-185-load-60.csv']

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
