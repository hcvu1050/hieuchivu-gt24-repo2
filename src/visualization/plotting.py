import sys
import os
import yaml
sys.path.append("..")

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_REPORT_FOLDER = os.path.join (ROOT_FOLDER, 'reports')
SOURCE_CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')

import pandas as pd
import matplotlib.pyplot as plt

def batch_plot_loss (model_name:str, folder_name: str, ylims: list = None):
    # get list of configs
    
    config_folder_path = os.path.join (SOURCE_CONFIG_FOLDER, folder_name)
    config_file_list = os.listdir (config_folder_path)
    # config_file_list = [f for f in config_file_list if f.startswith(model_name)]
    config_file_list = [os.path.join(config_folder_path, f) for f in config_file_list]
        
    # get list of train loss files
    train_loss_folder_path = os.path.join (SOURCE_REPORT_FOLDER, model_name, 'train_loss', folder_name)
    train_loss_file_list = os.listdir (train_loss_folder_path)
    train_loss_file_list = [os.path.join (train_loss_folder_path, f) for f in train_loss_file_list]

    # PLOTTING
    num_grid_rows = len (train_loss_file_list)
    plt.figure(figsize=(12, 5 * num_grid_rows)) 
    
    for grid in range (1, len(train_loss_file_list) + len(config_file_list)+1):
        plt.subplot (num_grid_rows, 2, grid)
        
        if grid % 2 == 1: 
            if ylims is not None: plt.ylim(ylims) 
            plot_loss (
                train_loss_file_list[int((grid-1)/2)],
                title = 'name'
            )
        else:
            plot_config(
                config_file_list[int(grid/2-1)],
                title=  config_file_list[int(grid/2-1)].split(sep = "\\")[-1]
                )
        
def plot_loss (filename: str, title: str):
    history_df = pd.read_csv(filename)

    epochs = range(1, len(history_df) + 1)
    training_loss = history_df['loss']
    validation_loss = history_df['val_loss']
    # plt.figure(figsize=(6, 5))

    # plt.ylim(.50, .65) 
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def plot_config (filename: str, title: str): 
    with open (filename, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    formatted_text = yaml.dump(config, default_flow_style=False, indent=4, sort_keys=False)
    # Display the formatted text
    plt.text(0.1, 0.5, formatted_text, fontsize=11, va='center', ha='left')
    # Turn off axis for this subplot
    plt.axis('off')
    # Add a title
    plt.title(title)
    # Save or display the figure
    plt.tight_layout()
    