import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task3( strength_type,vector_file,file_id,word_dir_path):

    vector_df = pd.read_csv(vector_file,header=None)
    vector_df = vector_df.rename(columns={0: "file_id", 1: "word", 2: "tf", 3:"tf_idf", 4: "tf_idf2"})
    filtered_df = vector_df.loc[vector_df['file_id'] == file_id]
    wrd_file = file_id + '.wrd'
    sensor_word_map = {}
    for line in open(word_dir_path + '/' + wrd_file).readlines():
        word = line.split('|')[1]
        word = word.split('[')[1]
        word = word.split(']')[0]
        sensor_id = int(line.split('|')[0].split(',')[1])
        strength = filtered_df[filtered_df['word'] == word][strength_type].values[0]
        if sensor_id in sensor_word_map:
            sensor_word_map[sensor_id].append(strength)
        else:
            sensor_word_map[sensor_id] = [strength]
    df = pd.DataFrame.from_dict(sensor_word_map,orient='index')
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    sns.heatmap(df, cmap=cmap)
    # plt.show()
    plt.savefig(word_dir_path + '/' + file_id + '_'+ strength_type + '.png')
    return df

import os,argparse
def file_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Task 3')

parser.add_argument('--indir', type=dir_path,help = 'input files directory')
parser.add_argument('--input', type=str,help = 'input file')
parser.add_argument('--strength',type=str,help =  'type of strength')
args = parser.parse_args()

dir_path = args.indir
task2_output_directory = dir_path + '/task2_output'
task1_output_directory = dir_path + '/task1_output'

task3(args.strength,task2_output_directory+ '/vectors.csv',args.input,task1_output_directory)