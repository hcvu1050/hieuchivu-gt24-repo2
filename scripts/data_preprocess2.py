import sys
import os
sys.path.append("..")
import argparse
from src.data.ingestion2 import collect_data

TRAIN_CV_TEST_RATIO = [.7,.15, .15]

def main():
    collect_data ()


if __name__ == '__main__':
    main()