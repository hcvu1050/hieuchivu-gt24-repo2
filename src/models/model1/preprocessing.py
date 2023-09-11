import os
import pandas as pd
import numpy as np
import tensorflow as tf
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
SOURCE_FILENAME = 'FINAL.txt'
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, SOURCE_FILENAME)

TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/processed')
TARGET_PREFIX = 'model1_'

INPUT_GROUP_LAYER_NAME = 'input_Group'
INPUT_TECHNIQUE_LAYER_NAME = 'input_Technique'
RANDOM_STATE = 13

def _get_data ():
    """ Get the necessary files from data/interim
    """
    #1 get files
    print ('Collecting data')
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
    
    train_files = [file_name for file_name in csv_file_names if 'train' in file_name]
    cv_files = [file_name for file_name in csv_file_names if 'cv' in file_name]
    test_files = [file_name for file_name in csv_file_names if 'test' in file_name]
    
    #2 read as DataFrame
    train_set = _get_Xy_dfs (train_files)
    cv_set = _get_Xy_dfs (cv_files)
    test_set = _get_Xy_dfs (test_files)
    return train_set, cv_set, test_set

def _get_Xy_dfs (file_list: list):
    """
    Finds and map file to input and output DataFrame
    """
    X_group_file_name =     [file_name for file_name in file_list if 'X_group' in file_name][0]
    X_technique_file_name = [file_name for file_name in file_list if 'X_technique' in file_name][0]
    y_file_name =           [file_name for file_name in file_list if '_y' in file_name][0]
    print (X_group_file_name)
    print (X_technique_file_name)
    print (y_file_name)

    
    X_group_df = pd.read_csv (os.path.join (SOURCE_PATH, X_group_file_name))
    X_technique_df = pd.read_csv (os.path.join (SOURCE_PATH, X_technique_file_name))
    y_df = pd.read_csv (os.path.join (SOURCE_PATH, y_file_name))
        
    return {
        'X_group' : X_group_df,
        'X_technique': X_technique_df,
        'y': y_df
    }

def _build_dataset(df_set, frac: float = None):
    """
    From a set(train, cv, test) containing X and y values: Create a a tensorflow Dataset
    """
    X_group = df_set['X_group'].drop(columns = 'group_ID')
    X_technique = df_set['X_technique'].drop(columns = 'technique_ID')
    y = df_set['y']
    
    
    X_group_tf = tf.convert_to_tensor(X_group.values, dtype = tf.float32)
    X_technique_tf = tf.convert_to_tensor(X_technique.values, dtype = tf.float32)
    y_tf = tf.convert_to_tensor(y.values, dtype = tf.float32)
    
    res_dataset = tf.data.Dataset.from_tensor_slices ((
        {
            INPUT_GROUP_LAYER_NAME: X_group_tf, 
            INPUT_TECHNIQUE_LAYER_NAME: X_technique_tf
            },
        y_tf))
    return res_dataset

# Define a function to serialize your data
# def _serialize_example(dictionary, float_value):
#     # Convert tensors to numpy arrays
#     dict_value = {key: value.numpy() for key, value in dictionary.items()}
#     float_value = float_value.numpy()
    
    
#     # Ensure the values are NumPy arrays with float32 data type
#     dict_value = {key: np.array(value, dtype='float32') for key, value in dict_value.items()}
#     float_value = np.array(float_value, dtype='float32')
    
#     feature = {
#         'input_Group': tf.train.Feature(float_list=tf.train.FloatList(value=dict_value['input_Group'])),
#         'input_Technique': tf.train.Feature(float_list=tf.train.FloatList(value=dict_value['input_Technique'])),
#         'float_value': tf.train.Feature(float_list=tf.train.FloatList(value=[float_value])),
#     }
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()

def _save_dataset (dataset, file_name):
    file_path = os.path.join (TARGET_PATH, file_name)
    tf.data.Dataset.save (dataset, file_path)
    print ('Dataset saved to', file_path)

def model_preprocess(partial_train: float = None):
    train_set, cv_set, test_set = _get_data()
    train_dataset = _build_dataset(train_set)
    
    element_spec = train_dataset.element_spec
    print (element_spec)
    dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
    print (dataset_size)
    
    _save_dataset (dataset = train_dataset, file_name= 'train_dataset')
    # cv_data_set = _build_dataset (cv_set)
    # test_data_set = _build_dataset(test_set)