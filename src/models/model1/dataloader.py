import os
import pandas as pd
import numpy as np
import tensorflow as tf

from .constants import ROOT_FOLDER, TRAIN_DATASET_FILENAME, CV_DATASET_FILENAME, TEST_DATASET_FILENAME, RANDOM_STATE
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/processed')

def _single_load (file_name):
    print ("loading\t", file_name)
    file_path = os.path.join (SOURCE_PATH, file_name) 
    dataset = tf.data.Dataset.load (file_path)
    print ('loaded:\t', file_name)
    print('DataSet size:\t', len(dataset), 'examples')
    return dataset

def load_data (sample_train: float = None):
    """
    sample_train: option to sample and train only a fraction of train_dataset
    """
    train_dataset = _single_load (TRAIN_DATASET_FILENAME)
    cv_dataset = _single_load (CV_DATASET_FILENAME)
    test_dataset = _single_load (TEST_DATASET_FILENAME)
    
    if sample_train is not None:
        num_samples = int(len(train_dataset) * sample_train)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=RANDOM_STATE).take(num_samples)
        
    return train_dataset, cv_dataset, test_dataset