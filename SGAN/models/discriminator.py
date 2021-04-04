#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa  # pip install tensorflow-addons
import numpy as np


DefaultDiscriminatorOpts = {
    "conv_kernel_size": 5,
    "padding": 'same',
    "strides": 2,
    "num_classes": 86,
}


class Discriminator(tf.keras.Model):
    """
    Stacked Discriminator Models With Shared Weights
    (supervised and unsupervised discriminator);
    Discriminate between real and fake, and try to classify the labeled data;
    6-layer Deep CNN with weight normalization and dropout
    """

    def __init__(self, opts: dict = DefaultDiscriminatorOpts):
        super(Discriminator, self).__init__()
        self._opts = opts

        # downsample
        #optional: self.input_layer = tf.keras.layers.InputLayer(input_shape=(32,224,224,3))

        self.conv1_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=64,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
            # input_shape=(224,224,3)
        )

        # downsample
        self.conv2_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=128,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv2 = layers.Conv2D(
            filters=128,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
        )

        # downsample
        self.conv3_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=256,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv3 = layers.Conv2D(
            filters=256,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
        )

        # downsample
        self.conv4_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=512,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv4 = layers.Conv2D(
            filters=512,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
        )

        # downsample
        self.conv5_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=512,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv5 = layers.Conv2D(
            filters=512,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
        )

        self.conv6_wn = tfa.layers.WeightNormalization(
            layers.Conv2D(
                filters=1024,
                kernel_size=self._opts["conv_kernel_size"],
                strides=self._opts["strides"],
                padding=self._opts["padding"],
                activation=None,
            ))

        self.conv6 = layers.Conv2D(
            filters=1024,
            kernel_size=self._opts["conv_kernel_size"],
            strides=self._opts["strides"],
            padding=self._opts["padding"],
            activation=None,
        )

        # prevent overfitting
        self.dropout = layers.Dropout(0.4)

        # features from discriminator
        # flatten via GAP
        # Compared to fully connected layers, higher robustness for spatial translation and less overfitting concerns
        self.pool = layers.GlobalAveragePooling2D()

        # neurons at the output layer
        self.out_layer_wn = tfa.layers.WeightNormalization(
            layers.Dense(
                units=opts["num_classes"],
                activation=None
            ))

        self.out_layer = layers.Dense(
            units=opts["num_classes"],
            activation=None
        )

    @tf.function
    def call(self, x, is_training):
        #x = self.input_layer(x)
        x = self.conv1_wn(x)
        #x = self.conv1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.dropout(x, training=is_training)
        x = self.conv2_wn(x)
        #x = self.conv2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv3_wn(x)
        #x = self.conv3(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.dropout(x, training=is_training)
        x = self.conv4_wn(x)
        #x = self.conv4(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv5_wn(x)
        #x = self.conv5(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.dropout(x, training=is_training)
        x = self.conv6_wn(x)
        #x = self.conv6(x)
        # output produced here will be used in feature matching loss; no dropout or bn
        x = tf.nn.leaky_relu(x, alpha=0.2)
        features = self.pool(x)
        # class_logits: inputs to softmax distribution over the different classes
        class_logits = self.out_layer_wn(features)
        #class_logits = self.out_layer(features)

        # supervised output - probabilities over classes
        # s_prob = tf.nn.softmax(class_logits)
        # unsupervised output
        # uns_prob = self.unsupervised_activation(class_logits)
        # The function below uses TF built-in function, more numerically stable than above - log(sum(exp(input))).
        # It avoids overflows caused by taking the exp of large inputs and underflows
        # caused by taking the log of small inputs.
        # gan_logits = tf.reduce_logsumexp(class_logits, 1)
        return features, class_logits

    # normalized sum of the exponential outputs for unsupervised discriminator model in the SGAN
    # class_logits here is the activation of neurons before softmax of the supervised part
    def unsupervised_activation(self, class_logits):
        # the probability of an input image being real == sum over the real class logits
        logexpsum = np.sum(np.exp(class_logits))
        # feed those values to LogSumExp that models the binary classification value to be fed into sigmoid function
        result = logexpsum / (logexpsum + 1.0)
        return result

    # model_opts will be saved for later reloading the model
    def model_opts(self):
        return self._opts


def discriminator_supervised_loss(labels_lab, logits_imgs_labeled):
    # if labels are int and not one-hot encoded, can use tf.nn.sparse_softmax_cross_entropy_with_logits()
    sup_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_lab, logits=logits_imgs_labeled))
    return sup_loss


def discriminator_unsupervised_loss(logits_real_unl, logits_fake):
    logits_sum_real = tf.reduce_logsumexp(logits_real_unl, axis=1)
    logits_sum_fake = tf.reduce_logsumexp(logits_fake, axis=1)
    unsup_loss = 0.5 * (
        tf.negative(tf.reduce_mean(logits_sum_real)) +
        tf.reduce_mean(tf.nn.softplus(logits_sum_real)) +
        tf.reduce_mean(tf.nn.softplus(logits_sum_fake)))
    # loss_real_unl = 0.5 * [ tf.negative(tf.reduce_mean(logits_sum_real)) + tf.reduce_mean(tf.nn.softplus(logits_sum_real)) ]
    # loss_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(logits_sum_fake))
    return unsup_loss

# loss for discriminator = loss for classification problem (supervised) + loss for GAN problem (unsupervised)
#                        = loss for classification problem (supervised) + loss_real_unl + loss_fake


def discriminator_loss(labels_lab, logits_imgs_labeled, logits_real_unl, logits_fake):
    d_loss = discriminator_supervised_loss(
        labels_lab, logits_imgs_labeled) + discriminator_unsupervised_loss(logits_real_unl, logits_fake)
    return d_loss


if __name__ == "__main__":
    input = tf.zeros((1, 224, 224, 3))

    discriminator_test = Discriminator(DefaultDiscriminatorOpts)
    s_output, uns_output = discriminator_test(input)

    print("s_output is ", s_output)
    print("uns_output is ", uns_output)
    discriminator_test.summary()
