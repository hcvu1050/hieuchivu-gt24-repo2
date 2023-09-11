import sys
import os
sys.path.append("..")
import argparse

from src.models.model1.preprocessing import model_preprocess

def main():
    model_preprocess()
    
if __name__ == '__main__':
    main()