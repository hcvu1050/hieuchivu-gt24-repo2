import sys
import os
sys.path.append("..")
import argparse
from src.data import ingestion, cleaning, build_features

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    ingestion.collect_data()
    cleaning.clean_data(target_path= TARGET_PATH)
    technique_features, group_features = build_features.build_features()
    
if __name__ == '__main__':
    main()