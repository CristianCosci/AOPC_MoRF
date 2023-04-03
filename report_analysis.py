import numpy as np
import pickle
import matplotlib.pyplot as plt

name1 = 'gradcam_results_8_28'
name2 = 'eigencam_results_8_28'

origin_results_path = 'original_results_8_28.pkl'

with open(origin_results_path, 'rb') as f:
    origin_results = pickle.load(f)

with open(name1 + '.pkl', 'rb') as f:
    result1 = pickle.load(f)

with open(name2 + '.pkl', 'rb') as f:
    result2 = pickle.load(f)


name_id = [i for i in result1.keys()]

# for id in name_id:
#     plt.plot(origin_results[id][1])
#     plt.savefig('AOPC_plots/original/AOPC_origin_img' + str(id) + '.png')
#     plt.clf()

#     plt.plot(result1[id][1])
#     plt.savefig('AOPC_plots/'+ name1 + '/AOPC_' + name1 + '_img' + str(id) + '.png')
#     plt.clf()

#     plt.plot(result2[id][1])
#     plt.savefig('AOPC_plots/'+ name2 + '/AOPC_' + name2 + '_img' + str(id) + '.png')
#     plt.clf()


# ------------------------------

# data = [[origin_results[i][2] for i in range(20,30)], [result1[i][2] for i in range(20,30)], [result2[i][2] for i in range(20,30)]]
# X = np.arange(20,30)

# plt.figure(figsize=(12,6))
# plt.bar(X - 0.25, data[0], color = 'green', width = 0.25)
# plt.bar(X + 0.00, data[1], color = 'red', width = 0.25)
# plt.bar(X + 0.25, data[2], color = 'blue', width = 0.25)
# plt.legend(labels=['original', name1, name2])
# plt.xticks(X)
# plt.savefig('AOPC_value_plot_20_30.png')

# ------------------------------------------

# to_sum_origin = 0
# to_sum_result1 = 0
# to_sum_result2 = 0

# for id in name_id:
#     to_sum_origin += origin_results[id][2]
#     to_sum_result1 += result1[id][2]
#     to_sum_result2 += result2[id][2]


# print('AOPC mean original img: ', to_sum_origin / (len(name_id)))
# print('AOPC mean ' + name1 + ' img: ', to_sum_result1 / (len(name_id)))
# print('AOPC mean ' + name2 + ' img: ', to_sum_result2 / (len(name_id)))


# ----------------------------------------

# diff_result1 = []
# diff_result2 = []
# for id in name_id:
#     diff_result1.append(origin_results[id][2] - result1[id][2])
#     diff_result2.append(origin_results[id][2] - result2[id][2])


# print('MAX drawdown using ' + name1 + ': ' , np.max(diff_result1), \
#       ' at index: ', diff_result1.index(np.max(diff_result1)))

# print('MAX drawdown using ' + name2 + ': ', np.max(diff_result2), \
#       ' at index: ', diff_result2.index(np.max(diff_result2)))

# print('MEAN drawdown using center_distance: ', np.mean(diff_result1))
# print('MEAN drawdown using ssim_not_inv: ', np.mean(diff_result2))

# ----------------------

# plt.plot(origin_results[4][1])
# plt.plot(result1[4][1])
# plt.legend(labels=['original', name1])
# plt.savefig('origin_center_distance_comparison_img4.png')