import tensorflow as tf
from utils import build_and_compile_cnn_model, prepare_train_datasets

BATCH_SIZE = 64
EPOCHS_NUM = 3

train_datasets = prepare_train_datasets(BATCH_SIZE)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_and_compile_cnn_model()
model.fit(x=train_datasets, epochs=EPOCHS_NUM)
