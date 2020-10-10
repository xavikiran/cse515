import os
import pandas as pd
import numpy as np
from scipy.integrate import quad
from window_slider import Slider
import json


def gaussianIntegrand(x):
    mu = 0.0
    sig = 0.25
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def calculateBandLength(i,r,full_area):
    return 2*quad(gaussianIntegrand, (i-r-1)/r, (i-r)/r)[0]/full_area

def get_quantized_number(x,band_lengths,r_resolution):
    if x==0:
        return r_resolution+1
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
    print("nan",x)


def get_file_ids(data_dir):
    file_ids = []
    for filename  in os.listdir(data_dir + '/W/'):
        if(filename.endswith('.csv')):
            file_ids.append(filename)
    return file_ids



def processGesture(file_id,data_dir):
    map = {}
    w_file =  data_dir + '/W/' + file_id
    x_file = data_dir + '/X/' + file_id
    y_file = data_dir + '/Y/' + file_id
    z_file = data_dir + '/Z/' + file_id
    map['W'] = processComponent(w_file,file_id,'W')
    map['X'] = processComponent(x_file, file_id, 'X')
    map['Y'] = processComponent(y_file, file_id, 'Y')
    map['Z'] = processComponent(z_file, file_id, 'Z')

    return map

def calcualate_average(list):
    if len(list) == 0:
        return 0
    sum = 0
    for i in list:
        sum+=i
    return sum/len(list)

def generate_average_amplitude(array,window,shift,bands,resolution):
    bucket_size = window
    overlap_count = window-shift
    slider = Slider(bucket_size, overlap_count)
    slider.fit(array)

    #change refactor
    map = {}
    symbolic = {}
    t=0
    while True:
        window_data = slider.slide()
        if len(window_data) == window :
            window_average = calcualate_average(window_data)
            map[t] = window_average
            symbolic[t] = get_quantized_number(window_average,bands,resolution)
            t+=shift

        if slider.reached_end_of_list(): break
    return map,symbolic

def generate_words(array,window,shift):
    bucket_size = window
    overlap_count = window-shift
    slider = Slider(bucket_size, overlap_count)
    slider.fit(array)

    #change refactor
    words = {}
    t=0
    while True:
        window_data = slider.slide()
        if len(window_data) == window :
            # words[t] = ','.join([str(i) for i in window_data])
            t+=shift
            words[t] = str(window_data)

        if slider.reached_end_of_list(): break
    return words

def processComponent(component_file,file_id,component):
    df = pd.read_csv(component_file, header=None).T
    # print(df)
    sensor_amplitudes = df.mean().tolist()
    sensor_deviations = df.std().tolist()
    sensor_id_amplitude_map = {}
    sensor_id_deviations_map = {}
    for i in range(0,len(sensor_amplitudes)):
        sensor_id_amplitude_map[i] = sensor_amplitudes[i]
        sensor_id_deviations_map[i] = sensor_deviations[i]
    #normalization
    df_norm = df.copy()
    for column in df_norm.columns:
        min = df_norm[column].min()
        max = df_norm[column].max()
        if min == max:
            df_norm[column].values[:] = 0
        else:
            df_norm[column] = (2*(df_norm[column] - min)) / (max - min) - 1

    #quantization
    full_area, err = quad(gaussianIntegrand, -1, 1)
    band_lengths = []
    for i in range(1, 2 * r_resolution + 1):
        length_band = calculateBandLength(i, r_resolution, full_area)
        band_lengths.append(length_band)

    quantized_df = df_norm.copy()
    for column in quantized_df:
        quantized_df[column] = quantized_df[column].apply(get_quantized_number,args=([band_lengths,r_resolution]))

    sensor_id_to_master_map = {}
    #words generation
    for column in quantized_df:
        words_map = generate_words(quantized_df[column].to_numpy(), w_window, s_shift)
        average_amplitude_map, symbolic_map = generate_average_amplitude(df_norm[column].to_numpy(), w_window, s_shift,
                                                                         band_lengths, r_resolution)
        master_map = {}
        # for key in words_map:
        #     master_map[key] = words_map[key], average_amplitude_map[key], symbolic_map[key]
        master_map['avg'] = sensor_id_amplitude_map[column]
        master_map['std'] = sensor_id_deviations_map[column]
        master_map['words'] = words_map
        master_map['symbolic'] = symbolic_map
        master_map['avg_amplitude'] = average_amplitude_map
        sensor_id_to_master_map[column] = master_map


    return sensor_id_to_master_map

data_dir = 'data/'
output_dir = 'output_task0/'
r_resolution = 4
w_window = 3
s_shift = 3

file_ids = get_file_ids(data_dir)
for file_id in file_ids:
    map = processGesture(file_id, data_dir)
    temp = json.dumps(map, indent=4)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = file_id.split('.')[0]
    output_file = output_dir + '/' + file_name + '.output'
    file_handle = open(output_file, 'w+', newline='')
    file_handle.write(temp)
    file_handle.close()
    with open(output_file) as json_file:
        data = json.load(json_file)


