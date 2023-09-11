import sys
import os
sys.path.append("..")
import argparse
import tensorflow as tf

from src.models.model1.preprocessing import model_preprocess
# from src.models.model1.dataloader import load_data

def main():
    model_preprocess()
    # dataset = load_data('train_dataset')
    # element_spec = dataset.element_spec
    # print (element_spec)
    
if __name__ == '__main__':
    main()