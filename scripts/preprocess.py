import sys
import os
sys.path.append("..")
import argparse
from src.data import ingestion, cleaning

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    ingestion.collect_data()
    cleaning.clean_data(target_path= TARGET_PATH)
    
if __name__ == '__main__':
    main()