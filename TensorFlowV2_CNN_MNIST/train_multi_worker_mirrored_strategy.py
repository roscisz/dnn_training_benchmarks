from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS_NUM = 3

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

print("\n\n\n***Preparing dataset***\n\n\n")


# Scaling MNIST data from (0, 255] to (0., 1.]
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)


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


print("\n\n\n***Training the model***\n\n\n")

train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

with strategy.scope():
    model = build_and_compile_cnn_model()
model.fit(x=train_datasets, epochs=EPOCHS_NUM)
