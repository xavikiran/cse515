import numpy as np
import pandas as pd
from scipy.integrate import quad

from window_slider import Slider
import csv,os,math,copy
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import ntpath
ntpath.basename("a/b/c")
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def normalize_data(filePath):
    df = pd.read_csv(filePath, header=None).T
    df_norm = (2*(df - df.min())) / (df.max() - df.min()) - 1  #normalization [-1,1]
    return df_norm

def gaussianIntegrand(x):
    mu = 0.0
    sig = 0.25
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calculateBandLength(i,r,full_area):
    return 2*quad(gaussianIntegrand, (i-r-1)/r, (i-r)/r)[0]/full_area

def get_quantized_number(x,band_lengths,r_resolution):
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
            file_output_line = str([file_id,sensor_id,t]) + '|' + str(list(window_data)) + '\n'
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


def task_1(file_id,file_path,output_file,r_resolution,w_window,s_shift):
    full_area, err = quad(gaussianIntegrand, -1, 1)
    band_lengths = []
    for i in range(1, 2 * r_resolution + 1):
        length_band = calculateBandLength(i, r_resolution, full_area)
        band_lengths.append(length_band)

    df = normalize_data(file_path)
    for column in df:
        df[column] = df[column].apply(get_quantized_number,args=([band_lengths,r_resolution]))
    quantized_df= df
    sensor_tf_map = {}
    file_handle = open(output_file,'w+',newline='')
    for column in quantized_df:
        sensor_word_tf_map = generate_words(quantized_df[column].to_numpy(), w_window, s_shift, file_id, column, output_file,file_handle)
        sensor_tf_map[column] = sensor_word_tf_map
    file_handle.close()

def run_task1_on_all_files(dir_path,r_resolution,w_window,s_shift,task1_output_directory):


    if not os.path.exists(task1_output_directory):
        os.makedirs(task1_output_directory)
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_id = filename.split('.')[0]
            output_file = task1_output_directory + '/' + file_id + '.wrd'
            task_1(file_id,dir_path + '/' + filename,output_file,r_resolution,w_window,s_shift)


import os
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Task 1')
parser.add_argument('--r', metavar='N', type=int, nargs='+',
                    help='a number for resolution')
parser.add_argument('--w', metavar='N', type=int, nargs='+',
                    help='a number for window')
parser.add_argument('--s', metavar='N', type=int, nargs='+',
                    help='a number for shift')
parser.add_argument('--indir', type=dir_path,help='input directory')



args = parser.parse_args()
dir_path = args.indir
task1_output_directory = dir_path + '/task1_output'

# dir_path = os.path.dirname(os.path.realpath(__file__))
run_task1_on_all_files(dir_path,args.r[0],args.w[0],args.s[0],task1_output_directory)


