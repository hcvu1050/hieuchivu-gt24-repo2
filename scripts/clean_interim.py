import sys
import os
sys.path.append("..")
# path to save the built-feature data
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
import shutil

def clear_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all files and subdirectories within the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Contents of {folder_path} have been cleared.")
        else:
            print(f"{folder_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    clear_folder(TARGET_PATH)