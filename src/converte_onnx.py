import keras, tf2onnx, logging
import tensorflow as tf

from src.models.layer import DecompositionLayer, FeatureWiseScalingLayer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pred_distance = 0
model_path = f'../outputs/checkpoints/model_{pred_distance}.keras'
model = keras.models.load_model(filepath=model_path, custom_objects={'DecompositionLayer': DecompositionLayer,
                                                                     'FeatureWiseScalingLayer': FeatureWiseScalingLayer})
logging.info(f'Model loaded from {model_path}')

spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
logging.info('converted ONNX model')

with open(f'../outputs/checkpoints/model_{pred_distance}.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

logging.info('saved ONNX model')
