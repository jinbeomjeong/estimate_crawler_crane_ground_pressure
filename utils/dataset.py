import os, cv2, pytesseract, re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
feature_names = ['swing_angle(deg)', 'boom_angle(deg)', 'weight(ton)']
target_names = ['fl_pressure(kg/cm^2)', 'fr_pressure(kg/cm^2)', 'rl_pressure(kg/cm^2)', 'rr_pressure(kg/cm^2)', 'left_pressure_length(m)', 'right_pressure_length(m)']


def img_to_dataset(img_path_list: list) -> np.array:
    dataset_length = len(img_path_list)
    target_status = np.zeros(shape=(dataset_length, len(target_names)), dtype=np.float64)
    feature_status = np.zeros(shape=(dataset_length, len(feature_names)), dtype=np.float64)

    for i, img_path in enumerate(tqdm(img_path_list, ncols=100, desc='data analyzing...')):
        feature_parameter = img_path[img_path.rfind(os.sep) + 1:].rstrip('.jpg')

        swing_angle = int(feature_parameter.split('swing-')[1].split('-boom')[0])
        boom_angle = int(feature_parameter.split('boom-')[1].split('-weight')[0])
        weight = int(feature_parameter.split('weight-')[1].split('-')[0])

        feature_status[i, 0] = swing_angle
        feature_status[i, 1] = boom_angle
        feature_status[i, 2] = weight

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fl_p_img = gray[628:646, 2220:2305]
        fr_p_img = gray[628:646, 2462:2545]
        rl_p_img = gray[814:830, 2220:2305]
        rr_p_img = gray[814:830, 2462:2545]

        pressure_imgz = [fl_p_img, fr_p_img, rl_p_img, rr_p_img]

        for i, pressure_img in enumerate(pressure_imgz):
            p_str = pytesseract.image_to_string(pressure_img)

            try:
                target_status[i, 0] = float(re.findall('\d+.\d+', p_str)[0])

            except:
                target_status[i, 0] = -1

        left_length_img = np.transpose(np.flip(gray[662:808, 2260:2280], axis=0))
        right_length_img = np.transpose(np.flip(gray[662:808, 2485:2505], axis=0))

        length_imgz = [left_length_img, right_length_img]

        for i, length_img in enumerate(length_imgz):
            length_str = pytesseract.image_to_string(length_img)

            try:
                target_status[i, 1] = float(re.findall('\d+.\d+', length_str)[0])

                if 6.75 < target_status[i, 1]:
                    target_status[i, 1] = target_status[i, 1] / 100
            except:
                target_status[i, 1] = -1

    return pd.DataFrame(np.hstack((feature_status, target_status)), columns=feature_names+target_names)


class Dataset:
    def __init__(self, file_path_list: list):
        self.__data_file_name_list = file_path_list
        self.__data_feature_names = ['Boom_Angle(deg)', 'Swing_Angle(deg)', 'Load(Ton)']
        self.__data_target_names = ['Actual_Load_Left_1(N)', 'Actual_Load_Left_2(N)', 'Actual_Load_Left_3(N)',
                                    'Actual_Load_Left_4(N)', 'Actual_Load_Left_5(N)', 'Actual_Load_Right_1(N)',
                                    'Actual_Load_Right_2(N)', 'Actual_Load_Right_3(N)', 'Actual_Load_Right_4(N)',
                                    'Actual_Load_Right_5(N)']
        self.__train_dataset = pd.DataFrame()
        self.__val_dataset = pd.DataFrame()
        self.__raw_dataset = self.load_dataset(file_path_list)
        self.__filtered_dataset = self.filtered_dataset(self.__raw_dataset)

        len_data = self.__filtered_dataset.shape[0]
        self.__n_of_interval = int(len_data / (len_data * 0.2))

    @staticmethod
    def get_load_value(data_file_name: str) -> int:
        re_iter = re.compile('[' + 'load' + ']+').finditer(data_file_name)
        load_idx = 0

        for i in re_iter:
            if i.group() == 'load':
                load_idx = i.end()
                load_idx += 1
                break

        load_str = data_file_name[load_idx:]
        load = int(load_str[0:load_str.find('-')])

        return load

    def filtered_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame():
        filtered_dataset = dataset.copy()

        filtered_dataset = filtered_dataset[filtered_dataset['Time(sec)'] > 2.0]

        for target_name in self.__data_target_names:
            filtered_dataset = filtered_dataset[filtered_dataset[target_name] > 1000]

        filtered_dataset['Boom_Angle(deg)'] = filtered_dataset['Boom_Angle(deg)'] + 70
        filtered_dataset.reset_index(inplace=True, drop=True)

        return filtered_dataset

    def load_dataset(self, data_file_names: list) -> pd.DataFrame():
        raw_dataset = pd.DataFrame()
        data_header = ['Time(sec)'] + self.__data_target_names + self.__data_feature_names

        for file_name in tqdm(data_file_names):
            raw = pd.read_csv('data' + os.sep + file_name, encoding='cp949', header=None)
            raw.drop(0, inplace=True)
            raw.columns = data_header
            raw.reset_index(inplace=True, drop=True)
            raw = raw.astype(np.float64)

            load = self.get_load_value(file_name)
            raw['Load(Ton)'] = np.full(shape=(raw.shape[0]), fill_value=load, dtype=np.float64)

            raw_dataset = pd.concat([raw_dataset, raw], axis=0)

        raw_dataset.reset_index(inplace=True, drop=True)

        return raw_dataset

    def get_dataset(self) -> pd.DataFrame:
        return self.__filtered_dataset

    def get_data_feature_names(self) -> list:
        return self.__data_feature_names

    def get_data_target_names(self) -> list:
        return self.__data_target_names

    def get_train_dataset(self) -> pd.DataFrame():
        train_dataset = self.__filtered_dataset.copy()
        train_dataset = train_dataset.drop(self.__filtered_dataset.index[::self.__n_of_interval])

        return pd.concat([train_dataset[self.__data_feature_names], train_dataset[self.__data_target_names]], axis=1)

    def get_val_dataset(self) -> pd.DataFrame():
        val_dataset = self.__filtered_dataset.copy()
        val_dataset = val_dataset[::self.__n_of_interval]

        return pd.concat([val_dataset[self.__data_feature_names], val_dataset[self.__data_target_names]], axis=1)
