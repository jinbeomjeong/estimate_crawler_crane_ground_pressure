import keras, tf2onnx, logging
import tensorflow as tf

from src.models.layer import DecompositionLayer, FeatureWiseScalingLayer, gelu_approximate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seq_len = 50
pred_distance = 30

model_path = f'../outputs/checkpoints/model_seq_{seq_len}_pred_{pred_distance}_15m.keras'
model = keras.models.load_model(filepath=model_path, custom_objects={'DecompositionLayer': DecompositionLayer,
                                                                     'FeatureWiseScalingLayer': FeatureWiseScalingLayer,
                                                                     'gelu_approximate': gelu_approximate})
logging.info(f'Model loaded from {model_path}')

spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
logging.info('converted ONNX model')

with open(f'../outputs/checkpoints/model_seq_{seq_len}_pred_{pred_distance}_15m.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

logging.info('saved ONNX model')
