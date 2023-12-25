import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, interp2d


boom_cal = pd.read_csv('data/boom_cal.csv')
crane_load = pd.read_csv('data/crane_load.csv')

load = boom_cal['Unnamed: 0']
load = load.rename('cal_load(ton)').to_numpy()
angle = np.array(boom_cal.columns[1:].astype(int), dtype=np.int32)
deform = boom_cal.to_numpy()[:, 1:]

f1 = interp2d(x=angle, y=load, z=deform, kind='linear')
f2 = interp1d(x=crane_load['boom_angle(deg)'], y=crane_load['work_range(m)'])
f3 = interp1d(x=crane_load['work_range(m)'], y=crane_load['load(ton)'])

input_angle = 61.9
input_load = 55.8

new_work_range = f2(input_angle).item()+(f1(input_angle, input_load).item()/1000)
new_load = f3(new_work_range).item()

print(new_work_range, new_load)
