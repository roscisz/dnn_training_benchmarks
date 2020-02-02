import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags


# Scaling MNIST data from (0, 255] to (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def prepare_train_datasets(batch_size):
    BUFFER_SIZE = 10000
    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)
    train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
    train_datasets = train_datasets_unbatched.batch(batch_size)
    return train_datasets


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model
