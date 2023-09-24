"""
last update: 2023-09-24
V2
training script for model1 - single train
the config files for training (one at a time) are stored in `configs/model1_single_config_train`
"""
import time
import sys
import os
import yaml
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse

sys.path.append("..")

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
### folder for config of single model train
SINGLE_TRAIN_FOLDER_NAME = 'model1_single_train'

REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports', 'model1')
TRAINED_MODELS_FOLDER = os.path.join (ROOT_FOLDER, 'trained_models')

from src.models.model1.preprocessing import model_preprocess
from src.models.model1.dataloader import load_data
from src.models.model1.model_v0_4 import Model1
def main():
    #### PARSING ARGS
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file to build and train the model')
    
    parser.add_argument ('-pp', '--preprocess', type = bool,
                         required= False, default= False,
                         help = 'option to preprocess the data first (only necessary for the first time)')
    args = parser.parse_args()
    config_file_name = args.config
    preprocess = args.preprocess

    #### LOAD CONFIGS FROM CONFIG FILE
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, SINGLE_TRAIN_FOLDER_NAME,config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
        
    model_architecture_config = config['model_architecture']
    train_config = config['train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    print ('---config for Model1\n',config)

    #### LOAD DATASETS, THEN CONFIG DATASETS
    if preprocess: model_preprocess(train_set_split= 0.7)
    train_dataset, train_cv_dataset, cv_dataset, test_dataset, feature_info = load_data(load_train_cv_set= True)
    
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


    train_cv_dataset = train_cv_dataset.batch(32)
    # test_dataset = test_dataset.batch(32)
    
    #### LOAD/COMPILE MODEL THEN TRAIN MODEL
    model = Model1 (input_sizes= feature_info,
                config=model_architecture_config)    
    
    optimizer = keras.optimizers.Adam (learning_rate= learning_rate)    
    loss = keras.losses.BinaryCrossentropy (from_logits= True)
    model.compile (optimizer, loss = loss)
    
    ## TRAIN MODEL ☝️ needs to add/fix more things if oversampling problem is true
    start_time = time.time()
    history = model.fit (
        train_dataset,
        validation_data= train_cv_dataset,
        epochs=epochs
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = elapsed_time  % 60
    print(f"Training completed in {elapsed_minutes} minutes and {elapsed_seconds} seconds")
    
    #### SAVE HISTORY
    history_df = pd.DataFrame(history.history)
    file_name = '{config_file}.csv'.format(config_file = args.config)
    file_path = os.path.join (REPORT_FOLDER, 'train_loss', SINGLE_TRAIN_FOLDER_NAME,file_name)
    history_df.to_csv(file_path, index=False)
    
    
    #### SAVE TRAINED MODEL: tk
    
if __name__ == '__main__':
    main()
    
    
