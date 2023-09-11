import sys
import os
sys.path.append("..")
import argparse
import tensorflow as tf

from src.models.model1.preprocessing import model_preprocess
from src.models.model1.dataloader import load_data

def main():
    model_preprocess()
    train_dataset, cv_dataset, test_dataset = load_data()
    element_spec = train_dataset.element_spec
    print (element_spec)
    element_spec = cv_dataset.element_spec
    print (element_spec)
    element_spec = test_dataset.element_spec
    print (element_spec)
    
if __name__ == '__main__':
    main()