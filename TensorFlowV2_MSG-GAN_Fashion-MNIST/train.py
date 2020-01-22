#!/usr/bin/env python
# coding: utf-8

# Based on: https://blog.paperspace.com/tensorflow-2-0-in-practice/


import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from tqdm import tqdm
from random import random


BATCH_SIZE = 32
NUM_EXAMPLES = 3
NOISE_DIM = 32


tf.__version__


#Function for reshaping images into the multiple resolutions we will use
def image_reshape(x):
    return [
        tf.image.resize(x, (7, 7)),
        tf.image.resize(x, (14, 14)),
        x
    ]


def mnist_dataset(batch_size):
    #fashion MNIST is a drop in replacement for MNIST that is harder to solve 
    (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
    train_images = train_images/127.5  - 1
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    dataset = dataset.map(image_reshape)
    dataset = dataset.cache()
    dataset = dataset.shuffle(len(train_images))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_DEVICES = strategy.num_replicas_in_sync
print(NUM_DEVICES)

dataset =  mnist_dataset(NUM_DEVICES * BATCH_SIZE)
dataset_distr = strategy.experimental_distribute_dataset(dataset)
print(dataset_distr)


def generator_model():
    outputs = []

    z_in = tf.keras.Input(shape=(NOISE_DIM,))
    x = layers.Dense(7*7*256)(z_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    for i in range(3):
        if i == 0:
            x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2),
                padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        outputs.append(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1),
            padding='same', activation='tanh')(x))

    model = tf.keras.Model(inputs=z_in, outputs=outputs)
    return model


def discriminator_model():
    # we have multiple inputs to make a real/fake decision from
    inputs = [
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.Input(shape=(14, 14, 1)),
        tf.keras.Input(shape=(7, 7, 1))
    ]

    x = None
    for image_in in inputs:
        if x is None:
            # for the first input we don't have features to append to
            x = layers.Conv2D(64, (5, 5), strides=(2, 2),
                padding='same')(image_in)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.3)(x)
        else:
            # every additional input gets its own conv layer then appended
            y = layers.Conv2D(64, (5, 5), strides=(2, 2),
                padding='same')(image_in)
            y = layers.LeakyReLU()(y)
            y = layers.Dropout(0.3)(y)
            x = layers.concatenate([x, y])

        x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    inputs = inputs[::-1] # reorder the list to be smallest resolution first
    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model


# create the models and optimizers for later functions
with strategy.scope():
    generator = generator_model()
    discriminator = discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

    # prediction of 0 = fake, 1 = real
    #@tf.function
    def discriminator_loss(real_output, fake_output):
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(real_output), real_output)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(fake_output), fake_output)
        return tf.nn.compute_average_loss(real_loss + fake_loss)

    #@tf.function
    def generator_loss(fake_output):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(fake_output), fake_output)
        return tf.nn.compute_average_loss(loss)

    #@tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        #gradient tapes keep track of all calculations done in scope and create the
        #    gradients for these
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            dis_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss,
            generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(dis_loss,
            discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator,
            generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
            discriminator.trainable_variables))

        return gen_loss, dis_loss

    @tf.function
    def distributed_train(images):
        gen_loss, dis_loss = strategy.experimental_run_v2(
            # remove the tf functions decorator for train_step
            train_step, args=(images,))
        # this reduces the return losses onto one device
        gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_loss, axis=None)
        dis_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dis_loss, axis=None)
        return gen_loss, dis_loss

    # also change train() to use distributed_train instead of train_step
    def train(dataset, epochs):
        for epoch in tqdm(range(epochs)):
            for image_batch in dataset:
                gen_loss, dis_loss = distributed_train(image_batch)


    train(dataset_distr, 10)
