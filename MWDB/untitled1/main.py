import task1 as t1
import task2 as t2
import task3 as t3
import task4 as t4
import sys
import numpy as np

r = sys.argv[1]
w = sys.argv[2]
s = sys.argv[3]
similarity = 'cosine'
dir_path = '/Users/xavi/MWDB/Phase_1/Z/'
task1_output_directory = dir_path + '/task1_output'
task2_output_directory = dir_path + 'task2_output'
# t1.run_task1_on_all_files(dir_path,3,3,2,task1_output_directory)
# t2.task2(dir_path,20,task2_output_directory)
# for file_id in ['test1','test2','test3','test4','test5','test6']:
#     t3.task3('tf',task2_output_directory+ '/vectors.csv',file_id,task1_output_directory)
#     t3.task3('tf_idf',task2_output_directory+ '/vectors.csv',file_id,task1_output_directory)
#     t3.task3('tf_idf2',task2_output_directory+ '/vectors.csv',file_id,task1_output_directory)
keys = t4.task4('test1',task2_output_directory+ '/vectors.csv','tf',dir_path,similarity)
# print(keys)
# keys = t4.task4('test6',task2_output_directory+ '/vectors.csv','tf_idf',dir_path,similarity)
# print(keys)
# keys = t4.task4('test3',task2_output_directory+ '/vectors.csv','tf_idf2',dir_path,similarity)
# print(keys)
from sklearn.metrics.pairwise import cosine_similarity
