import joblib, time
import numpy as np

from tqdm.auto import tqdm


est_model = joblib.load('rf_model.joblib')
time_list = []

for i in tqdm(range(1000)):
    t0 = time.time()
    pred_output = est_model.predict(np.random.rand(1, 15).astype(np.float32))
    time_list.append(time.time() - t0)

print(np.mean(time_list))
