#!/usr/bin/env python3

# quick start:  python SGAN_train.py --data_dir /mnt/2ndSSD/chimpanzee_faces-master/datasets_cropped_chimpanzee_faces --annotation_dir /mnt/2ndSSD/chimpanzee_faces-master/Freytag_dataset_splits/compiled_skf/ --model_dir /mnt/2ndSSD/chimpanzee_faces-master/models/ --summary_dir /mnt/2ndSSD/chimpanzee_faces-master/summary/ --max_train_epochs 200 --batch_size 32
# quick start: python SGAN_train.py --data_dir /home/am/Rachael/chimpanzee_faces-master/datasets_cropped_chimpanzee_faces --annotation_dir /home/am/Rachael/chimpanzee_faces-master/Freytag_dataset_splits/compiled_skf/ --model_dir /home/am/Rachael/chimpanzee_faces-master/models/ --summary_dir /home/am/Rachael/chimpanzee_faces-master/Freytag_dataset_splits/summary/ --max_train_epochs 200 --batch_size 32
import os
import sys
import argparse
import datetime

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as backend
import numpy as np

from models.generator import Generator, DefaultGeneratorOpts, generate_fake_samples, generator_loss
from models.discriminator import Discriminator, DefaultDiscriminatorOpts, discriminator_loss, discriminator_supervised_loss
from data.facedataset import (
    CsvCompile,
    CsvSplit,
    LoadDatasets,
    LoadUnlabeled,
)
from utils.summary import SummaryWriter

parser = argparse.ArgumentParser()

""" Directory parameters """
parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the root directory of image datasets.",
)

parser.add_argument(
    "--annotation_dir",
    type=str,
    help="The path to the annotation (csv files) directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

""" Training parameters """

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the classifier."
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

ARGS = parser.parse_args()

"""
Control for preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":

    generatorOpts = DefaultGeneratorOpts
    discriminatorOpts = DefaultDiscriminatorOpts

    # in this identity/name prediction experiment: 86
    num_classes = discriminatorOpts["num_classes"]
    batch_size = ARGS.batch_size
    n_epochs = ARGS.max_train_epochs
    latent_dim = generatorOpts["latent_dim"]

    gen_optimizer = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    dis_optimizer = tf.keras.optimizers.Adam(lr=0.002 * 0.1, beta_1=0.5)

    # 5-fold cross-validation
    splits = CsvSplit(ARGS.annotation_dir, 5, 'name')
    # return defaultdict(<class 'list'>, {1: ('../train_1.csv', '../test_1.csv'), 2: ..);
    folds = splits.load()
    # Each split in each fold has 86 distinct names/IDs of animals

    for i in range(1, 6):
        backend.clear_session()

        generator = Generator(generatorOpts)
        discriminator = Discriminator(discriminatorOpts)

        summary_writer = SummaryWriter(ARGS.summary_dir)
        train_summary_writer, test_summary_writer = summary_writer.get_writer()

        train, test = folds[i]
        # normalize(-1,1), resize, one-hot encoding of labels, shuffle, batch, prefetch take place here
        prepare_train = LoadDatasets(train)
        train_ds = prepare_train.load_ds(batch_size)

        prepare_test = LoadDatasets(test)
        test_ds = prepare_test.load_ds(batch_size)

        unlabeled_csv = os.path.join(
            ARGS.annotation_dir, 'unlabeled_woVideo.csv')
        prepare_unlabeled = LoadUnlabeled(unlabeled_csv)
        imgs_real_unl_ds = prepare_unlabeled.load_unlabeled_ds(batch_size)

        # Define random noise to use for generating 8 images for inspection.
        seed = tf.random.normal(shape=[8, latent_dim])

        step = 0

        for epoch in range(n_epochs):
            # train starts here
            imgs_real_unl_ds_iter = iter(imgs_real_unl_ds)
            print("Training at epoch:", str(epoch))
            for (imgs_labeled, targets) in train_ds:
                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

                    imgs_fake = generate_fake_samples(
                        generator, latent_dim, batch_size)  # num_samples = batch_size

                    logits_real_unl, features_real_unl = discriminator(
                        imgs_real_unl_ds_iter.get_next(), is_training=True)
                    logits_fake, features_fake = discriminator(
                        imgs_fake, is_training=True)

                    _, logits_imgs_labeled = discriminator(
                        imgs_labeled, is_training=True)

                    # calculate loss and gradients, apply updates, write to summary
                    # gen_loss is feature matching less (minimize L2 distance of
                    # the average features of the generated images vs. the average features of the real images
                    gen_loss = generator_loss(features_real_unl, features_fake)

                    # (targets, logits_imgs_labeled): input to calculate loss of supervised discriminator
                    # (logits_real_unl, logits_fake): input to calc loss of unsupervised discriminator
                    dis_loss = discriminator_loss(
                        targets, logits_imgs_labeled, logits_real_unl, logits_fake)

                    gen_grad = generator_tape.gradient(
                        gen_loss, generator.trainable_variables)
                    dis_grad = discriminator_tape.gradient(
                        dis_loss, discriminator.trainable_variables)

                    gen_optimizer.apply_gradients(
                        zip(gen_grad, generator.trainable_variables))
                    dis_optimizer.apply_gradients(
                        zip(dis_grad, discriminator.trainable_variables))

                    with train_summary_writer.as_default():
                        tf.summary.scalar(
                            'fold{}_generator_loss'.format(i), gen_loss, step=step)
                        tf.summary.scalar(
                            'fold{}_discriminator_loss'.format(i), dis_loss, step=step)

                    print('Epoch_{}_Step_{}'.format(epoch, step),
                          'fold{}_generator_loss:'.format(i), gen_loss.numpy())
                    print('Epoch_{}_Step_{}'.format(epoch, step),
                          'fold{}_discriminator_loss:'.format(i), dis_loss.numpy(),)

                    step += 1

            # test starts here
            # at the end of each epoch, test the loss of the accuracy of supervised discriminator
            print("Testing after epoch - ", str(epoch), ":")

            test_losses = []
            test_accuracies = []

            test_step = 0
            for (input, target) in test_ds:
                _, logits_imgs_labeled = discriminator(input, False)
                sample_test_loss = discriminator_supervised_loss(
                    target, logits_imgs_labeled)
                print('sample_test_loss is', sample_test_loss.numpy())

                prediction = tf.nn.softmax(logits_imgs_labeled)
                
                # axis=0 is the batch_size, axis=1 is the one-hot encoding of the labels
                sample_test_accuracy = np.argmax(
                    target, axis=1) == np.argmax(prediction, axis=1)

                test_losses.append(sample_test_loss.numpy())
                # compute the sample_test_accuracy for each batch before appending
                test_accuracies.append(np.mean(sample_test_accuracy))

            test_dis_sup_loss = np.mean(test_losses)
            test_classification_accuracy = np.mean(test_accuracies)

            with test_summary_writer.as_default():
                tf.summary.scalar('test_step_{}_fold_{}_test_dis_sup_loss'.format(
                    test_step, i), test_dis_sup_loss, step=epoch)
                tf.summary.scalar('test_step_{}_fold_{}_test_accuracy_dis_sup'.format(
                    test_step, i), test_classification_accuracy, step=epoch)

            print('Epoch: ', str(epoch), 'Test loss of supervised discriminator:', test_dis_sup_loss,
                  'Test accuracy of supervised classifier:', test_classification_accuracy)

            fake = generator(seed, is_training=False)
            with test_summary_writer.as_default():
                tf.summary.image('fold{}_fake_images'.format(
                    i), (fake + 1) / 2, step=epoch, max_outputs=8)
