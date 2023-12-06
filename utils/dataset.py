import os, cv2, pytesseract, re
import numpy as np
import pandas as pd
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
feature_names = ['swing_angle(deg)', 'boom_angle(deg)', 'weight(ton)']
target_names = ['fl_pressure(kg/cm^2)', 'fr_pressure(kg/cm^2)', 'rl_pressure(kg/cm^2)', 'rr_pressure(kg/cm^2)', 'left_pressure_length(m)', 'right_pressure_length(m)']


def load_dataset(img_path_list: list) -> np.array:
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
