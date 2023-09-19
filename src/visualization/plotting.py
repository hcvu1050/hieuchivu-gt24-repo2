import sys
import os
import yaml
sys.path.append("..")

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_REPORT_PATH = os.path.join (ROOT_FOLDER, 'reports')
SOURCE_CONFIG_PATH = os.path.join (ROOT_FOLDER, 'configs')

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss (filename: str, title: str):
    history_df = pd.read_csv(filename)

    epochs = range(1, len(history_df) + 1)
    training_loss = history_df['loss']
    validation_loss = history_df['val_loss']
    # plt.figure(figsize=(6, 5))

    plt.ylim(.50, .65) 
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def plot_config (filename: str, title: str): 
    with open (filename, 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    formatted_text = yaml.dump(config, default_flow_style=False, indent=4)
    # Display the formatted text
    plt.text(0.1, 0.5, formatted_text, fontsize=12, va='center', ha='left')
    # Turn off axis for this subplot
    plt.axis('off')
    # Add a title
    plt.title(title)
    # Save or display the figure
    plt.tight_layout()
    