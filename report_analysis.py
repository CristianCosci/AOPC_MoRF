import numpy as np
import pickle
import matplotlib.pyplot as plt

# AOPC results filename
filename = 'results'

with open(filename + '.pkl', 'rb') as f:
    result = pickle.load(f)

name_id = [i for i in result.keys()]

# Example of plotting AOPC curve for each image
for id in name_id:
    plt.plot(result[id][1])
    plt.savefig('AOPC_plots/AOPC_' + filename + '_img' + str(id) + '.png')
    plt.clf()


to_sum_result = 0

for id in name_id:
    to_sum_result += result[id][2]

# Print mean AOPC value across all the image
AOPC_name = to_sum_result / len(name_id)
print('AOPC mean ' + filename + ' img: ', AOPC_name)