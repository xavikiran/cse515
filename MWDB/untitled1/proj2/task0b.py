import copy
import math
import os,json
import pandas as pd
import numpy as np
import sklearn.decomposition.pca
from scipy.linalg import svd

from os import path

from sklearn import preprocessing
from sklearn.decomposition import PCA,TruncatedSVD

task0_output_dir = 'output_task0/'
data_dir = 'data/'
task1_output_dir = 'output_task1/'
if not os.path.exists(task1_output_dir):
    os.makedirs(task1_output_dir)

def get_file_ids(data_dir):
    file_ids = []
    for filename  in os.listdir(data_dir + '/W/'):
        if(filename.endswith('.csv')):
            file_ids.append(filename)
    return file_ids

def read_task0a_output(file):
    data = json.load(open(file))
    return data

def calcualte_tf(file_id,task0_output_dir):
    file_name = file_id.split('.')[0] + '.output'
    map = read_task0a_output(task0_output_dir + file_name)
    tf_dictionary = {}
    word_count = 0
    for component in ['W', 'X', 'Y', 'Z']:
        component_map = map[component]
        for sensor_id in component_map:
            sensor_data = component_map[sensor_id]
            for value in sensor_data['words'].values():
                word_count += 1
                dict_key = component + '-' + str(sensor_id) + '-' + str(value)
                if dict_key in tf_dictionary:
                    tf_dictionary[dict_key] = tf_dictionary[dict_key] + 1
                else:
                    tf_dictionary[dict_key] = 1
    n = word_count
    for key in tf_dictionary:
        tf_dictionary[key] = tf_dictionary[key]/n
    return tf_dictionary

def caluculate_idf(file_tf_map):
    N = len(file_tf_map.keys())
    idf_map = {}
    for key in file_tf_map:
        tf_map = file_tf_map[key]
        for word in tf_map:
            if word in idf_map:
                idf_map[word] +=1
            else:
                idf_map[word] = 1

    for key,value in idf_map.items():
        idf_map[key] = math.log(N/float(value))
    return idf_map

file_ids = get_file_ids(data_dir)
file_tf_map = {}
for file_id in file_ids:
    tf_map = calcualte_tf(file_id,task0_output_dir)
    file_tf_map[file_id] = tf_map

file_tf_idf_map = copy.deepcopy(file_tf_map)
idf_map = caluculate_idf(file_tf_map)

for file,word_map in file_tf_map.items():
    print(file + "|",word_map)
    for word,tf in word_map.items():
        file_tf_idf_map[file][word] = tf* idf_map[word]

print("\n")
print(idf_map)
print(file_tf_idf_map['1.csv'])
df = pd.DataFrame.from_dict(file_tf_map, orient='index')
df = df.fillna(0)
for column in df.columns:
    print(column)
temp  = df.std(axis=1)
pd.set_option('precision', 19)
print(temp)
# calcualte_tf('1.csv',task0_output_dir)
ndarr = df.T.to_numpy()[0]
np.savetxt("foo.csv", ndarr, delimiter=",")
def pca(feature_matrix,k):

    pca = PCA(n_components=k)
    principalComponents = pca.fit_transform(feature_matrix)
    print(pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(data=principalComponents)
    data_scaled = pd.DataFrame(preprocessing.scale(df), columns=df.columns)
    word_score_df = pd.DataFrame(pca.components_, columns=data_scaled.columns)
    print(word_score_df.iloc[0].sort_values(ascending=False))
    # print(principalDf)
    return 0

def svd(feature_matrix,k):

    svd = TruncatedSVD(n_components=k)
    components = svd.fit_transform(feature_matrix)
    print(svd.explained_variance_ratio_)
    principalDf = pd.DataFrame(data=components)
    # print(principalDf)
    return 0

pca(df,5)

# svd(df,5)

