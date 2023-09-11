import os
import pandas as pd
import numpy as np
import tensorflow as tf

from .constants import ROOT_FOLDER, TRAIN_DATASET_FILENAME, CV_DATASET_FILENAME, TEST_DATASET_FILENAME
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/processed')

def _single_load (file_name):
    print ("loading dataset")
    file_path = os.path.join (SOURCE_PATH, file_name) 
    dataset = tf.data.Dataset.load (file_path)
    return dataset

def load_data ():
    train_dataset = _single_load (TRAIN_DATASET_FILENAME)
    cv_dataset = _single_load (CV_DATASET_FILENAME)
    test_dataset = _single_load (TEST_DATASET_FILENAME)
    return train_dataset, cv_dataset, test_dataset