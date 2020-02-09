import tensorflow as tf
from utils import build_and_compile_cnn_model, prepare_train_datasets
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch size', lower_bound=0)
flags.DEFINE_integer('epochs', 3, 'Number of epochs', lower_bound=0)
flags.DEFINE_integer('steps_per_epoch', 500, 'Number of steps in each epoch', lower_bound=0)


def main(argv):
    train_datasets = prepare_train_datasets(FLAGS.batch_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_and_compile_cnn_model()
    model.fit(x=train_datasets, epochs=FLAGS.epochs, steps_per_epoch=FLAGS.steps_per_epoch)


if __name__ == '__main__':
    app.run(main)
