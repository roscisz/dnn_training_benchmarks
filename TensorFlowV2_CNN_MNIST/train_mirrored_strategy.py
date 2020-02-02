import tensorflow as tf
import tensorflow_datasets as tfds
from utils import scale, build_and_compile_cnn_model

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPOCHS_NUM = 3

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_and_compile_cnn_model()
model.fit(x=train_datasets, epochs=EPOCHS_NUM)
