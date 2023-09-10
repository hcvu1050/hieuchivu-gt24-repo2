import os
import pandas as pd
import tensorflow as tf
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'model1_'

def _get_data ():
    """ Get the necessary files from data/interim
    """
    