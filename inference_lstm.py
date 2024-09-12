import os, time
import numpy as np
import onnxruntime as ort

model_path = 'model.onnx'
model = ort.InferenceSession(model_path)

input_name = model.get_inputs()[0].name  # 첫 번째 입력 이름 (입력이 여러 개일 경우 각각 확인)
input_shape = model.get_inputs()[0].shape  # 입력 텐서의 모양 확인
input_type = model.get_inputs()[0].type  # 입력 타입 확인

print(input_name, input_shape, input_type)

input_data = np.random.randn(1, 30, input_shape[-1]).astype(np.float32)

pred = np.zeros(1)

for i in range(100):
    t0 = time.time()
    pred = model.run(None, {input_name: input_data})[0][0]
    print(f"Iteration {i+1}: {(time.time() - t0)*1000:.2f} ms")

print("Output:", pred)

