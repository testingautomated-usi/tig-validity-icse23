import tensorflow as tf
import numpy as np

def preprocess_tfds(imagenet_val, img_dim):
    x_test = imagenet_val.map(lambda x, _: x).prefetch(tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test.map(lambda x: tf.image.resize(x, [img_dim, img_dim]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test_resized.map(lambda x: x / np.float32(255),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test_resized = x_test_resized.prefetch(tf.data.experimental.AUTOTUNE)
    y_test_iter = imagenet_val.map(lambda _, y: y).as_numpy_iterator()
    y_test = np.fromiter(y_test_iter, dtype=int)
    return x_test_resized, y_test


def deprocess_image_tf(x):
    x = x.reshape((224, 224, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x