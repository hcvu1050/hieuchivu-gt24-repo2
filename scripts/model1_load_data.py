import sys
import os
sys.path.append("..")
import argparse
import tensorflow as tf

from src.models.model1.dataloader import load_data

def main():
    ## parsing arguments
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('--sample_train','-st', 
                         type = float, 
                         default= None, 
                         help='optional argument to sample a traction of train_dataset for training')
    args = parser.parse_args()
    sample_train = args.sample_train
    
    load_data (sample_train= sample_train)
    
if __name__ == '__main__':
    main()
