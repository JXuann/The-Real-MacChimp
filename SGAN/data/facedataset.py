#!/usr/bin/env python3

import os
import sys
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

import tensorflow as tf

from collections import defaultdict

csv_header = "filename,name,age,age_group,gender,right_eye,left_eye,mouth_center,left_ear_lobe,right_ear_lobe"
# change the root dir to yours
#imgs_root_directory = '/mnt/2ndSSD/chimpanzee_faces-master/datasets_cropped_chimpanzee_faces'
imgs_root_directory = '/home/am/Rachael/chimpanzee_faces-master/datasets_cropped_chimpanzee_faces'


class CsvCompile():
    """
    Compile annotations across datasets with files listed as excluded being filtered
    """

    def __init__(
        self,
        working_dir: str
    ):
        self.csv_header = csv_header
        self.working_dir = working_dir

    def _check_in_line(self, line, excluded_filenames):
        return any([id in line for id in excluded_filenames])

    def _filter_inputs(self, path_filelist_imgs: str, path_excluded: str, path_filtered: str):
        excluded_filenames = []
        with open(path_excluded, "r") as f_in:
            for line in f_in:
                excluded_filenames.append(line.rstrip('\n'))
        with open(path_filelist_imgs, "r") as f_in, open(path_filtered, "a") as f_out:
            for line in f_in:
                if not self._check_in_line(line, excluded_filenames):
                    f_out.write(line)

    def _write_csv(self, f_in, f_out):
        with open(f_in, "r") as f_in, open(f_out, "a") as f_out:
            for line in f_in:
                f_out.write(line)

    def __call__(self, first_ds_name, sec_ds_name, first_exc_list=None, sec_exc_list=None):
        print('start compiling')
        csv_file_path = os.path.join(self.working_dir, "compiled.csv")
        with open(csv_file_path, "a+") as f_out:
            f_out.write(self.csv_header)
        if first_exc_list is not None:
            self._filter_inputs(first_ds_name, first_exc_list, csv_file_path)
        else:
            self._write_csv(first_ds_name, csv_file_path)
        if sec_exc_list is not None:
            self._filter_inputs(sec_ds_name, sec_exc_list, csv_file_path)
        else:
            self._write_csv(sec_ds_name, csv_file_path)
        print("finished compiling")
        return csv_file_path


class CsvSplit():

    def __init__(
            self,
            working_dir: str,
            k_fold: int,
            target: str,  # target could be name, gender, age, age_group etc. In this pilot study, the prediction is name
    ):
        self.working_dir = working_dir
        self.k_fold = k_fold
        self.target = target
        self.skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        # key: fold_no; value: list of train-test (each fold) tuples
        self.splits = defaultdict(list)
        self.split_opt = ["train", "test"]

    """
    Return split for a required split (i.e. train/dev/test).
    If there is already a CSV split existing, return this split.
    If there is no split yet, generate a CSV split.
    A file named 'compiled.csv' needs to be provided under working_dir for skf.
    """

    def load(self):
        if self.can_load_from_csv():
            for fold_no in range(1, self.k_fold + 1):
                self.splits[fold_no] = (
                    os.path.join(self.working_dir, "train_" +
                                 str(fold_no) + ".csv"),
                    os.path.join(self.working_dir, "test_" +
                                 str(fold_no) + ".csv")
                )
            return self.splits

        else:
            df = pd.read_csv(os.path.join(self.working_dir, 'compiled.csv'))
            data = df.loc[:, 'filename']
            target = df.loc[:, self.target]
            fold_no = 1
            for train_idx, test_idx in self.skf.split(data, target):
                train = df.loc[train_idx, :]
                test = df.loc[test_idx, :]
                train_fn = 'train_' + str(fold_no) + '.csv'
                test_fn = 'test_' + str(fold_no) + '.csv'
                train_path = os.path.join(self.working_dir, train_fn)
                test_path = os.path.join(self.working_dir, test_fn)
                train.to_csv(train_path, index=False)
                test.to_csv(test_path, index=False)
                self.splits[fold_no] = (train_path, test_path)
                fold_no += 1
            return self.splits

    def can_load_from_csv(self):
        # presence of all csv split files need to be checked - could be improved later
        skf_file = os.path.join(self.working_dir, 'train_1.csv')
        if os.path.isfile(skf_file):
            print('can load existing splits')
            return True
        else:
            print('cannot load')
            return False


def read_image(img_path: str):
    directory = imgs_root_directory
    img_file = tf.io.read_file(directory + '/' + img_path)
    # default 0: use the number of channels in the PNG-encoded image
    img_tensor_uint8 = tf.io.decode_png(img_file)
    img_tensor_float32 = tf.cast(img_tensor_uint8, tf.float32)
    return img_tensor_float32


def onehot_labels(labels):
    # encode labels to integer
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)

    # encode labels to one_hot
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded_labels = integer_encoded_labels.reshape(
        len(integer_encoded_labels), 1)
    onehot_encoded_labels = onehot_encoder.fit_transform(
        integer_encoded_labels)
    return onehot_encoded_labels


def normalize_0_1(img):
    max = 255
    min = 0
    result = tf.math.subtract(img, min)
    norm_image = tf.math.divide(result, (max - min))
    return norm_image

# The output of the generator uses a hyperbolic tangent function, meaning its output is in [-1, 1].
# Therefore, the data needs to be rescaled in that range as well before feeding into discriminator


def normalize_neg_1(img):
    result = normalize_0_1(img)
    result = tf.math.multiply(result, 2)
    norm_image = tf.math.subtract(result, 1)
    return norm_image


def resize(img, target_h=224, target_w=224):
    # resize with 0-padding
    img = tf.image.resize_with_pad(
        img,
        target_h,
        target_w,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    return img


class LoadDatasets():
    def __init__(self, split):
        self.df = pd.read_csv(split)

    def load_ds(self, batch_size):
        targets = self.df['name'].values
        ds_targets = tf.data.Dataset.from_tensor_slices(onehot_labels(targets))

        img_paths = self.df['filename'].values
        ds_imgs = tf.data.Dataset.from_tensor_slices(img_paths)
        ds_imgs = ds_imgs.map(read_image)
        ds_imgs = ds_imgs.map(normalize_neg_1)
        ds_imgs = ds_imgs.map(resize)

        ds = tf.data.Dataset.zip((ds_imgs, ds_targets))
        ds = ds.shuffle(buffer_size=200)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds


class LoadUnlabeled():
    def __init__(self, unlabeled_csv):
        self.df = pd.read_csv(unlabeled_csv)

    def load_unlabeled_ds(self, batch_size):
        imgs_unl_paths = self.df['filename'].values
        imgs_unl_ds = tf.data.Dataset.from_tensor_slices(imgs_unl_paths)
        imgs_unl_ds = imgs_unl_ds.map(read_image)
        imgs_unl_ds = imgs_unl_ds.map(normalize_neg_1)
        imgs_unl_ds = imgs_unl_ds.map(resize)

        imgs_unl_ds = imgs_unl_ds.shuffle(buffer_size=200)
        imgs_unl_ds = imgs_unl_ds.batch(batch_size)
        imgs_unl_ds = imgs_unl_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return imgs_unl_ds


if __name__ == "__main__":
    # images = get_images_from_dir()
    # compiled = CsvCompile('/mnt/2ndSSD/chimpanzee_faces-master/tmp')
    # compiled_path = compiled(
    #    '/mnt/2ndSSD/chimpanzee_faces-master/tmp/ctai.csv',
    #    '/mnt/2ndSSD/chimpanzee_faces-master/tmp/czoo.csv',
    #    '/mnt/2ndSSD/chimpanzee_faces-master/tmp/ctai_exc.csv',
    # )

    splits = CsvSplit(
        '/mnt/2ndSSD/chimpanzee_faces-master/Freytag_dataset_splits/compiled_skf', 5, 'name')
    folds = splits.load()
    print(folds)
    train, test = folds[1]
    print(train)
