import os
import pandas as pd
# Get the root directory of the project
FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get collected data
TARGET_PATH = os.path.join (FILE_DIR, 'data/interim')

def _make_file_list (filename: str, target_path, content: list):
    """
    """
    # file_path = os.path.join(data_folder, filename)
    os.makedirs (target_path, exist_ok= True)
    if not filename.endswith ('.txt'): filename += '.txt'
    output_file = os.path.join (target_path, filename)
    
    with open (output_file, 'w') as file:
        for line in content:
            file.write (line)
            file.write ('\n')
    print ('List of exported files saved at: ', output_file)
    

def save_df_to_csv (df: pd.DataFrame, target_path, filename, prefix = '', postfix = '', output_list_file = None):
    os.makedirs (target_path, exist_ok = True)
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    filename = prefix + filename + postfix
    
    if not filename.endswith (".csv"): filename+= ".csv"
    output_file = os.path.join(target_path, filename)
    print ('Saving:\t', filename)
    df.to_csv (output_file, index = False)
    print ("Saved:\t",  filename, sep = '')
    print ("Finished: file saved to", target_path)
    
    if output_list_file is not None:
        _make_file_list (filename= filename, target_path=target_path, content= [filename])

def batch_save_df_to_csv (file_name_dfs: dict, target_path, prefix ='', postfix ='',output_list_file = None):
    """
    Saves the DataFrames stored in a dict as csv file. \n
    file_name_dfs: \n
        key: filename\n
        value = DataFrame\n
    prefix: a string added before filename
    """
    if not (prefix == '') and (not prefix.endswith ('_')): prefix += '_'
    if not (postfix == '') and (not prefix.startswith ('_')): postfix = '_' + postfix
    content = []
    
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key + postfix
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        df = file_name_dfs[key]
        print ('Saving:\t', filename)
        df.to_csv (output_file, index = False)
        content.append (filename)
        print ('Saved:\t',  filename, sep = '')
    
    print ("Finished: {} files saved to {}".format (len(file_name_dfs.keys()),target_path))
    
    if output_list_file is not None:
        # make a txt file containing the names of exported file
        _make_file_list (filename = output_list_file, target_path=target_path, content=content)
