import numpy as np
import pandas as pd
from scipy.integrate import quad
from window_slider import Slider
import csv,os,math,copy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import ntpath
import argparse
ntpath.basename("a/b/c")
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def generateVectorFile(final_vector_output, output_file_path):
    file_handle = open(output_file_path,'+w', newline='')
    for file,word_map in final_vector_output.items():
        for word,list in word_map.items():
            line = file + ',' + '\"' + word + '\"' + ',' + str(list[0]) +  ',' + str(list[1]) + ',' + str(list[2]) + '\n'
            file_handle.write(line)
    file_handle.close()

def task2(dir_path,max_sensors_count,task2_output_directory):
    file_tf_map = {}
    file_tf_count_map = {}
    file_sensor_idf2_map = {}

    task1_output_directory = dir_path + '/task1_output'
    for filename in os.listdir(task1_output_directory):
        if filename.endswith(".wrd"):
            file_id = filename.split('.')[0]
            output_file =  task1_output_directory + '/' + filename
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


    if not os.path.exists(task2_output_directory):
        os.makedirs(task2_output_directory)
    output_file = task2_output_directory + '/vectors.csv'
    generateVectorFile(final_vector_output,output_file)
    with open('idf_dataset.pickle', 'wb') as handle:
        pickle.dump(file_idf_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    return counter_map,tf_map,sensor_word_count_map

import os
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Task 2')

parser.add_argument('--indir', type=dir_path,help = 'input files directory')



args = parser.parse_args()
dir_path = args.indir
task2_output_directory = dir_path + '/task2_output'
task2(dir_path,20,task2_output_directory)