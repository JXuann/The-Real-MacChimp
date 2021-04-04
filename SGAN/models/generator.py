#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras.layers as layers


DefaultGeneratorOpts = {
    "latent_dim": 2048,
    "convT_kernel_size": 5,
    "padding": 'same',
    "strides": 2,
}


class Generator(tf.keras.Model):
    """
    6-layer Deep CNN with batch normalization
    """

    def __init__(self, opts: dict = DefaultGeneratorOpts):
        super(Generator, self).__init__()
        self._opts = opts

        self.dense = layers.Dense(
            units=7 * 7 * 256,
            activation=None,
            use_bias=False,
        )

        self.bn = layers.BatchNormalization()

        # prevent overfitting
        self.dropout = layers.Dropout(rate=0.4)

        # upsample to (14,14,128)
        self.convT1 = layers.Conv2DTranspose(
            filters=128,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()

        # upsample to (28,28,128)
        self.convT2 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()

        # upsample to (56,56,128)
        self.convT3 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            use_bias=False
        )
        self.bn3 = layers.BatchNormalization()

        # upsample to (112,112,128)
        self.convT4 = layers.Conv2DTranspose(
            filters=16,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            use_bias=False
        )
        self.bn4 = layers.BatchNormalization()

        # upsample to (224,224,128)
        self.convT5 = layers.Conv2DTranspose(
            filters=8,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            use_bias=False
        )
        self.bn5 = layers.BatchNormalization()

        # output [-1,1]
        # generate samples of size (224, 224, 3))
        self.out_layer = layers.Conv2DTranspose(
            filters=3,
            kernel_size=self._opts["convT_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation="tanh",
            use_bias=False
        )

    @tf.function
    def call(self, x, is_training):
        x = self.dense(x)
        x = self.bn(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = tf.reshape(x, shape=(-1, 7, 7, 256))
        x = self.dropout(x)

        x = self.convT1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.convT2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.convT3(x)
        x = self.bn3(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.convT4(x)
        x = self.bn4(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.out_layer(x)

        return x

# generate points in latent space as input for generator


def generate_latent_points(latent_dim, n_samples):
    noise = tf.random.normal(shape=[n_samples, latent_dim])
    return noise

# use generator to generate n fake examples, with class labels being 0


def generate_fake_samples(generator, latent_dim, n_samples):
    noise = generate_latent_points(latent_dim, n_samples)
    fake = generator(noise, True)
    # create class labels
    #y = zeros((n_samples, 1))
    return fake

# generator_loss is set to the 'feature matching' loss proposed in the 2016 paper of T. Salimans
# It consists of minimizing the absolute difference between the expected features on the real data and those on the generated samples
# It helps to train a stronger classifier than traditional GAN losses


def generator_loss(real_features, fake_features):
    real_moments = tf.reduce_mean(real_features, axis=0)
    fake_moments = tf.reduce_mean(fake_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(real_moments - fake_moments))
    return g_loss


if __name__ == "__main__":
    input = tf.random.normal([1, 2048])

    generator_test = Generator(DefaultGeneratorOpts)
    result = generator_test(input, False)

    print(result.shape)
    generator_test.summary()
