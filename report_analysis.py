import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('origin_results.pkl', 'rb') as f:
    origin_results = pickle.load(f)

with open('center_distance_results.pkl', 'rb') as f:
    center_distance_results = pickle.load(f)

with open('ssim_not_inv_results.pkl', 'rb') as f:
    ssim_not_inv_results = pickle.load(f)


name_id = [i for i in center_distance_results.keys()]

# for id in name_id:
#     plt.plot(origin_results[id][1])
#     plt.savefig('AOPC_plots/original/AOPC_origin_img' + str(id) + '.png')
#     plt.clf()

#     plt.plot(center_distance_results[id][1])
#     plt.savefig('AOPC_plots/center_distance/AOPC_center_distance_img' + str(id) + '.png')
#     plt.clf()

#     plt.plot(ssim_not_inv_results[id][1])
#     plt.savefig('AOPC_plots/ssim_not_inv/AOPC_ssim_not_inv_img' + str(id) + '.png')
#     plt.clf()


# ------------------------------


data = [[origin_results[i][2] for i in range(20,30)], [center_distance_results[i][2] for i in range(20,30)], [ssim_not_inv_results[i][2] for i in range(20,30)]]
X = np.arange(20,30)

plt.figure(figsize=(12,6))
plt.bar(X - 0.25, data[0], color = 'green', width = 0.25)
plt.bar(X + 0.00, data[1], color = 'red', width = 0.25)
plt.bar(X + 0.25, data[2], color = 'blue', width = 0.25)
plt.legend(labels=['original', 'center_distance', 'ssim_not_inv'])
plt.xticks(X)
plt.savefig('AOPC_value_plot_20_30.png')

# ------------------------------------------

# to_sum_origin = 0
# to_sum_center_distance = 0
# to_sum_ssim_not_inv = 0

# for id in name_id:
#     to_sum_origin += origin_results[id][2]
#     to_sum_center_distance += center_distance_results[id][2]
#     to_sum_ssim_not_inv += ssim_not_inv_results[id][2]


# print('AOPC mean original img: ', to_sum_origin / len(name_id))
# print('AOPC mean center_distance img: ', to_sum_center_distance / len(name_id))
# print('AOPC mean ssim_not_inv img: ', to_sum_ssim_not_inv / len(name_id))


# ----------------------------------------

# diff_center_distance = []
# diff_ssim_not_inv = []
# for id in name_id:
#     diff_center_distance.append(origin_results[id][2] - center_distance_results[id][2])
#     diff_ssim_not_inv.append(origin_results[id][2] - ssim_not_inv_results[id][2])


# print('MAX drowdown using center_distance: ', np.max(diff_center_distance), \
#       ' at index: ', diff_center_distance.index(np.max(diff_center_distance)))

# print('MAX drowdown using ssim_not_inv: ', np.max(diff_ssim_not_inv), \
#       ' at index: ', diff_ssim_not_inv.index(np.max(diff_ssim_not_inv)))

# print('MEAN drowdown using center_distance: ', np.mean(diff_center_distance))
# print('MEAN drowdown using ssim_not_inv: ', np.mean(diff_ssim_not_inv))

# ----------------------

# plt.plot(origin_results[4][1])
# plt.plot(center_distance_results[4][1])
# plt.legend(labels=['original', 'center_distance'])
# plt.savefig('origin_cente_distance_comparison_img4.png')