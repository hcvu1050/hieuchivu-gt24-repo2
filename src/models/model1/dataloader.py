import os
import pandas as pd
import numpy as np
import tensorflow as tf

from .constants import ROOT_FOLDER
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/processed')

def _single_load (file_name):
    print ("loading data")
    file_path = os.path.join (SOURCE_PATH, file_name) 
    dataset = tf.data.Dataset.load (file_path)
    return dataset

# def load_data (file_name):
    