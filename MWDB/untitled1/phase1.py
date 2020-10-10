import numpy as np
import pandas as pd
from scipy.integrate import quad
from window_slider import Slider
import csv,os,math,copy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import defaultdict, Counter

def normalize_data(filePath):
    df = pd.read_csv(filePath, header=None).T
    df_norm = (2*(df - df.min())) / (df.max() - df.min()) - 1  #normalization [-1,1]
    return df_norm

r_resolution = 3
w_window = 3
s_shift = 2
max_sensors_count = 20

def gaussianIntegrand(x):
    mu = 0.0
    sig = 0.25
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calculateBandLength(i,r,full_area):
    return 2*quad(gaussianIntegrand, (i-r-1)/r, (i-r)/r)[0]/full_area

def get_quantized_number(x,band_lengths):
    if x == -1:
        return 1
    elif x < 0 :
        x = abs(x)
        sum = 0
        for i in range(int(len(band_lengths)/2-1), -1,-1):
            if x >= sum and x < sum + band_lengths[i]:
                return i+1
            sum+=band_lengths[i]
    elif x > 0 and x != 1 :
        sum = 0
        for i in range(int(len(band_lengths)/2),len(band_lengths)):
            if x >= sum and x < sum + band_lengths[i]:
                return i+1
            sum += band_lengths[i]
    elif x ==1:
        return 2*r_resolution

#generate words

def generate_words(array,window,shift,file_id,sensor_id,output_file,file_handle):
    file = file_handle
    bucket_size = window
    overlap_count = window-shift
    slider = Slider(bucket_size, overlap_count)
    slider.fit(array)

    sensor_word_tf_map = {}
    #change refactor
    t=0
    while True:
        window_data = slider.slide()
        if len(window_data) == window :
            file_output_line = str([file_id,sensor_id,t]) +  '|' +  str(list(window_data)) + '\n'
            file.write(file_output_line)
            t+=shift
            word = str(list(window_data)).split('[')[1]
            word = word.split(']')[0]
            if word in sensor_word_tf_map:
                sensor_word_tf_map[word] += 1
            else:
                sensor_word_tf_map[word] = 1
        if slider.reached_end_of_list(): break
    return sensor_word_tf_map


def calculate_tf(file_path,max_sensors_count):
    file = open(file_path, 'r', newline='')
    counter_map = dict()
    sensor_word_count_map = {}
    for i in range(0,max_sensors_count):
        sensor_word_count_map[i] = {}
    word_count = 0
    for x in file:
        sensor_id = int(x.split('|')[0].split(',')[1])
        word = x.split('|')[1]
        word = word.split('[')[1]
        word = word.split(']')[0]
        counter_map[word] = counter_map.get(word, 0) + 1
        if word in sensor_word_count_map[sensor_id]:
            sensor_word_count_map[sensor_id] [word] +=1
        else:
            sensor_word_count_map[sensor_id][word] = 1
        word_count = word_count + 1
    tf_map = {}
    for freq in counter_map:
        tf_map[freq] = counter_map[freq]/word_count

    print(counter_map)
    return counter_map,tf_map,sensor_word_count_map


import ntpath
ntpath.basename("a/b/c")
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



def task_1(file_id,file_path,output_file):
    full_area, err = quad(gaussianIntegrand, -1, 1)
    band_lengths = []
    for i in range(1, 2 * r_resolution + 1):
        length_band = calculateBandLength(i, r_resolution, full_area)
        band_lengths.append(length_band)

    df = normalize_data(file_path)
    for column in df:
        df[column] = df[column].apply(get_quantized_number,args=([band_lengths]))
    quantized_df= df
    sensor_tf_map = {}
    file_handle = open(output_file,'w+',newline='')
    for column in quantized_df:
        sensor_word_tf_map = generate_words(quantized_df[column].to_numpy(), w_window, s_shift, file_id, column, output_file,file_handle)
        sensor_tf_map[column] = sensor_word_tf_map
    file_handle.close()


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
    print(N)
    return idf_map



def replaceWordByStrength(word,vectors,strength_type):
    t = vectors[word][strength_type]
    return t

def task3( strength_type,vector_file,file_id):

    vector_df = pd.read_csv(vector_file,header=None)
    vector_df = vector_df.rename(columns={0: "file_id", 1: "word", 2: "tf", 3:"tf_idf", 4: "tf_idf2"})
    filtered_df = vector_df.loc[vector_df['file_id'] == int(file_id)]
    wrd_file = file_id + '.wrd'
    sensor_word_map = {}
    for line in open(wrd_file).readlines():
        word = line.split('|')[1]
        word = word.split('[')[1]
        word = word.split(']')[0]
        sensor_id = int(line.split('|')[0].split(',')[1])
        print(sensor_id)
        strength = filtered_df[filtered_df['word'] == word][strength_type].values[0]
        if sensor_id in sensor_word_map:
            sensor_word_map[sensor_id].append(strength)
        else:
            sensor_word_map[sensor_id] = [strength]
    df = pd.DataFrame.from_dict(sensor_word_map,orient='index')
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    sns.heatmap(df, cmap=cmap)
    plt.show()
    return df

def generateVectorFile(final_vector_output, output_file_path):
    file_handle = open(output_file_path,'+w', newline='')
    for file,word_map in final_vector_output.items():
        for word,list in word_map.items():
            line = file + ',' + '\"' + word + '\"' + ',' + str(list[0]) +  ',' + str(list[1]) + ',' + str(list[2]) + '\n'
            file_handle.write(line)
    file_handle.close()

def euclidean(x):
    return (x[0] - x[1])**2

def calculateSimilarity(input_file_id,file_id,vector_df,strength):
    filtered_df = vector_df.loc[vector_df['file_id'] == int(file_id)]
    input_df = vector_df.loc[vector_df['file_id'] == int(input_file_id)]
    joined_df = pd.merge(filtered_df,input_df,how='outer',on='word')
    joined_df = joined_df[[strength + '_x',strength + '_y']]
    joined_df = joined_df.fillna(0)
    output_column = joined_df.apply(euclidean,axis=1)
    sum = output_column.sum()
    sum = sum**0.5
    return sum

def task4(input_file_id, vector_output,strength,dir_path):
    vector_df = pd.read_csv(vector_output, header=None)
    vector_df = vector_df.rename(columns={0: "file_id", 1: "word", 2: "tf", 3: "tf_idf", 4: "tf_idf2"})
    nearest_neighbours = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_id = filename.split('.')[0]
            similarity = calculateSimilarity(input_file_id,file_id,vector_df,strength)
            nearest_neighbours[file_id] = similarity
    nearest_neighbours = {k: v for k, v in sorted(nearest_neighbours.items(), key=lambda item: item[1])}
    nearest_neighbours.pop(str(input_file_id))
    keys = list(nearest_neighbours.keys())[:10]
    return keys

def runProgram(dir_path):
    file_tf_map = {}
    file_tf_count_map = {}
    file_sensor_idf2_map = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_id = filename.split('.')[0]
            output_file = file_id + '.wrd'
            task_1(file_id,dir_path + '/' + filename,output_file)
    #task2
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_id = filename.split('.')[0]
            output_file = file_id + '.wrd'
            count_map,tf_map,sensor_word_count_map = calculate_tf(output_file,max_sensors_count)
            sensor_idf2_map = caluculate_idf(sensor_word_count_map)
            file_sensor_idf2_map[file_id] = sensor_idf2_map
            file_tf_count_map[file_id] = count_map
            file_tf_map[file_id] = tf_map

    file_idf_map = caluculate_idf(file_tf_count_map)
    file_tf_idf_map = copy.deepcopy(file_tf_map)
    file_tf_idf2_map = copy.deepcopy(file_sensor_idf2_map)
    final_vector_output = {}
    for file,word_map in file_tf_map.items():
        final_vector_output[file] = {}
        for word,tf in word_map.items():
            file_tf_idf_map[file][word] = tf* file_idf_map[word]
            file_tf_idf2_map[file][word] = tf*file_sensor_idf2_map[file][word]
            final_vector_output[file][word] = [tf,file_tf_idf_map[file][word],file_tf_idf2_map[file][word]]

    for i in sorted(final_vector_output.keys()):
        print(i,final_vector_output[i])
    generateVectorFile(final_vector_output,'vectors.csv')
    with open('idf_dataset.pickle', 'wb') as handle:
        pickle.dump(file_idf_map, handle, protocol=pickle.HIGHEST_PROTOCOL)



    #task2 ends

    # task3('tf_idf2','vectors.csv','1')
    # task4(1,'vectors.csv','tf',dir_path)

def generate_vectors_for_query_file(query_file_path):
    file_id = path_leaf(query_file_path)
    file_id = file_id.split('.')[0]
    output_file = file_id + '.wrd'
    task_1(file_id,query_file_path,output_file)
    with open('idf_dataset.pickle', 'rb') as handle:
        file_idf_map = pickle.load(handle)
    #read query wrd file
    counter_map,tf_map,sensor_word_count_map = calculate_tf(output_file,max_sensors_count)
    sensor_idf2_map = caluculate_idf(sensor_word_count_map)
    return None

runProgram('/Users/xavi/MWDB/Phase_1/Z/')
# generate_vectors_for_query_file('/Users/xavi/MWDB/Phase_1/Z/test1.csv')