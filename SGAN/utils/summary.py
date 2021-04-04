#!/usr/bin/env python3

import os
import argparse
import datetime

import tensorflow as tf

class SummaryWriter():

    def __init__(self, summary_dir):
        self.summary_dir = summary_dir
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = "Chimp-face-ID-SGAN-{}".format(self.current_time)

    def get_writer(self):
        tensorboard_path = os.path.join(self.summary_dir, self.name)
        train_log_dir = os.path.join(
            tensorboard_path, 'gradient/' + self.current_time + '/train')
        test_log_dir = os.path.join('gradient/' + self.current_time + '/test')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer
