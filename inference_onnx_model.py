import time
import onnxruntime as ort
import numpy as np


model = ort.InferenceSession('models/model.onnx')
print('ONNX model loaded successfully.')

input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

print(f'Input name: {input_name}')
print(f'Output name: {output_name}')

a = np.full(shape=(30, 1), fill_value=70, dtype=np.float32)
b = np.full(shape=(30, 1), fill_value=0, dtype=np.float32)
input_buf = np.expand_dims(np.concatenate([a, b, b],axis=1), axis=0)


t0 = time.time()

while True:
    ts = time.time()

    input_buf = np.roll(input_buf, shift=-1, axis=1)
    #input_buf[:, -1, :] = input_data

    val_pred = np.squeeze(model.run(output_names=None, input_feed={'input': input_buf})).item()
    print(f'{time.time()-t0:.2f}', f'{time.time() - ts:.2f}', val_pred)


