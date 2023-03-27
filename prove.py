import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('origin_imgs/best_img0_.png')

img2 = cv2.imread('center_distance_imgs/best_img0_.png')

print('shape')
print(img1.shape)
print(img2.shape)


print('medie')
print(np.mean(img1))
print(np.mean(img2))

print('diff')
print(np.sum(img1[:,:,0] - img2[:,:,0]))
print(np.sum(img1[:,:,1] - img2[:,:,1]))
print(np.sum(img1[:,:,2] - img2[:,:,2]))
print(np.sum(img1 - img2))
# def perturbation():
#     pass

# # print(img.shape)
# # print('max value', np.max(img))
# block_size = 16
# blocks = []
# blocks_dict = {}
# key = 0
# verbose = False

# # Save block in a list and in a dict associated with the key
# for i in range(0, img.shape[0], block_size):
#     for j in range(0, img.shape[1], block_size):
#         block = img[i:i+block_size, j:j+block_size]
#         block_sum = np.sum(block)
#         blocks_dict[key] = (block, block_sum)
#         blocks.append(block)
#         key += 1

# if verbose:
#     fig, axs = plt.subplots(int(np.ceil(len(blocks)/14)), 14, figsize=(14,14))

#     # visualizza i blocchi come subplot
#     for i, block in enumerate(blocks):
#         row = i // 14
#         col = i % 14
#         axs[row, col].imshow(block, cmap='gray')
#         axs[row, col].axis('off')

#     # mostra la griglia di subplot
#     fig.savefig('prova.png')

#     img_reconstructed = np.vstack([
#         np.hstack(blocks[row*14:(row+1)*14])
#         for row in range(14)
#     ])

#     # visualizza l'immagine ricostruita
#     plt.imsave('elle.png', img_reconstructed, cmap='gray')

# if verbose:
#     for key, value in blocks_dict.items():
#         print(f"Chiave: {key}")
#         print(f"Array del blocco: {value}")


# sorted_blocks = sorted(blocks_dict.items(), key=lambda x: x[1][1], reverse=True)
# print(sorted_blocks[0][1][1])