import numpy as np
import os

source_video_path = './resources/Videos'
speaker_list = next(os.walk(source_video_path))[1]

word_index = 0
dict = {}
path = './resources/WordAlign'
word_list = np.load(path + '/speaker_output_test1')
for word in word_list:
    if word not in dict:
        if word != "sil":
            dict[word] = word_index
            word_index +=1

for i in range(6,12):
    output_vector = []
    word_list = np.load(path + '/speaker_output_test'+str(i))
    for j in range(len(word_list)):
        cur_vector = [0] * len(dict)
        cur_vector[dict[word_list[j]]] = 1
        output_vector.append(cur_vector)
    output_vector = np.asarray(output_vector)
    np.save(path + '/speaker_final_output_test'+str(i),output_vector)


for i in range(6,12):
    output_vector = []
    word_list = np.load(path +'/speaker_output_train'+str(i))
    for j in range(len(word_list)):
        cur_vector = [0] * len(dict)
        cur_vector[dict[word_list[j]]] = 1
        output_vector.append(cur_vector)
    output_vector = np.asarray(output_vector)
    np.save(path + '/speaker_final_output_train'+str(i),output_vector)
