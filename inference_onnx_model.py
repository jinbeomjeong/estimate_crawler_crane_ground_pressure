import time
import onnxruntime as ort
import numpy as np

from tqdm.auto import tqdm

model = ort.InferenceSession('rf_model.onnx')
print('ONNX model loaded successfully.')

input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

print(f'Input name: {input_name}')
print(f'Output name: {output_name}')

input_arr = np.random.rand(1, 15).astype(np.float32)

time_list = []

for i in tqdm(range(1000)):
    t0 = time.time()
    pred = model.run(output_names=None, input_feed={'float_input': input_arr})[0][0]
    time_list.append(time.time() - t0)

print(np.mean(time_list))
