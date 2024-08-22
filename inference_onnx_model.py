import time
import onnxruntime as ort
import numpy as np

model = ort.InferenceSession('rf_model.onnx')
print('ONNX model loaded successfully.')

input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

print(f'Input name: {input_name}')
print(f'Output name: {output_name}')

input_arr = np.random.rand(1, 3).astype(np.float32)
t0 = time.time()
pred = model.run(output_names=None, input_feed={'float_input': input_arr})[0][0]
print(f'Execution time: {time.time() - t0} seconds')
print(pred)
