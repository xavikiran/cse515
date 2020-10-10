

import os
import pandas as pd
from scipy.spatial.distance import cosine
def euclidean(x):
    return (x[0] - x[1])**2

def calculateSimilarity(input_file_id,file_id,vector_df,strength,similarity):
    filtered_df = vector_df.loc[vector_df['file_id'] == file_id]
    input_df = vector_df.loc[vector_df['file_id'] == input_file_id]
    joined_df = pd.merge(filtered_df,input_df,how='outer',on='word')
    col1 = strength + '_x'
    col2 = strength + '_y'
    joined_df = joined_df[[col1,col2]]
    joined_df = joined_df.fillna(0)
    if similarity == 'euclidean':
        output_column = joined_df.apply(euclidean,axis=1)
        sum = output_column.sum()
        sum = sum**0.5
        return sum
    elif similarity == 'cosine':
        sim = cosine(joined_df[col1], joined_df[col2])
        return sim

def task4(input_file_id, vector_output,strength,dir_path,similarity):
    vector_df = pd.read_csv(vector_output, header=None)
    vector_df = vector_df.rename(columns={0: "file_id", 1: "word", 2: "tf", 3: "tf_idf", 4: "tf_idf2"})
    nearest_neighbours = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_id = filename.split('.')[0]
            distance = calculateSimilarity(input_file_id,file_id,vector_df,strength,similarity)
            nearest_neighbours[file_id] = distance
    nearest_neighbours = {k: v for k, v in sorted(nearest_neighbours.items(), key=lambda item: item[1])}
    # nearest_neighbours.pop(input_file_id)
    print(nearest_neighbours)
    keys = list(nearest_neighbours.keys())[:11]
    return keys

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
parser.add_argument('--similarity',type=str,help =  'type of similarity')
args = parser.parse_args()

dir_path = args.indir
task2_output_directory = dir_path + '/task2_output'
task1_output_directory = dir_path + '/task1_output'

keys  = task4(args.input,task2_output_directory+ '/vectors.csv',args.strength,dir_path,args.similarity)
print(keys)