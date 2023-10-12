import os
import glob
import re
import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='AOPC MoRF')
parser.add_argument('imgs_dir', help='directory which contains images for AOPC')
parser.add_argument('cams_dir', help='directory which contains cams for AOPC')
parser.add_argument('-block_size', '--size', help='block size for evaluating AOPC', default=8)
parser.add_argument('-block_row', '--row', help='number of blocks per row in images', default=28)
parser.add_argument('-percentile', '--pct', help='percentile until compute AOPC', default=None)
parser.add_argument('-results_file_name', '--file_name', help='name of the file in which save results', default='results.pkl')
parser.add_argument('-v', '--verbose', help='enable verbose mode', default=False)
args = parser.parse_args()

# Imgs and Cams directories
directory_imgs = args.imgs_dir  #'imgs_new/'
directory_cams = args.cams_dir  #'cams_new/'
verbose = args.verbose

images_extension = ['*.png'] # possible images extensions
pattern_imgs = r'img_(\d+)\.png$' # regex pattern that corresponds to image names
pattern_cams = r'img_cam_(\d+)\.png$' # regex pattern that corresponds to cams names
imgs = []
cams = []

for estensione in images_extension:
    imgs.extend(sorted(glob.glob(os.path.join(directory_imgs, estensione)), key=lambda x: int(re.search(pattern_imgs, x).group(1))))
    cams.extend(sorted(glob.glob(os.path.join(directory_cams, estensione)), key=lambda x: int(re.search(pattern_cams, x).group(1))))

imgs_names = [os.path.basename(img) for img in imgs]
cams_names = [os.path.basename(img) for img in cams]

if verbose:
    print(imgs_names)
    print(cams_names)

# Image preprocessing
def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        # Resize((224, 224)),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


# Perturbation by block values mean
def perturbation(img):
    perturbation_value = []
    for z in range(img.shape[2]):
        perturbation_value.append(np.mean(img[:,:,z]))

    new_img = np.ones(img.shape)
    for z in range(img.shape[2]):
        new_img[:,:,z] = new_img[:,:,z] * perturbation_value[z]

    return new_img

# -----------------------------------------------------------------------------------

# PyTorch init and Model loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model = resnet50(pretrained=True).to(DEVICE)
model.eval()

total_AOPC = dict()
block_size = int(args.size)
block_per_row = int(args.row)
pct = int(args.pct) if args.pct is not None else -1

# Run for each images
for name_id in range(len(imgs_names)):
    img = cv2.imread(directory_cams+cams_names[name_id], cv2.IMREAD_GRAYSCALE)
    blocks_dict = {}
    key = 0

    # Save block in a list and in a dict associated with the key
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i+block_size, j:j+block_size] # Get block
            block_sum = np.sum(block)                   # Evaluate block with sum
            blocks_dict[key] = (block, block_sum)       # Save in dict
            key += 1
            
    # Sorting the blocks basing on Evaluation   
    sorted_blocks = sorted(blocks_dict.items(), key=lambda x: x[1][1], reverse=True)
    total_sum_blocks = sum(block_sum[1] for block_sum in list(blocks_dict.values()))  # Total sum of value by all blocks in img

    img_pred = cv2.imread(directory_imgs+imgs_names[name_id])[:,:,::-1]
    img_pred = np.float32(img_pred) / 255.0
    img_pred_aopc = img_pred.copy()

    # Classification on image -> f(x) at time 0 (zero perturbation)
    input_tensor = preprocess_image(img_pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)

    probs_k0, indices = torch.topk(torch.nn.Softmax(dim=-1)(output), k=1) # Get Label for first class and its probability score -> f(x) for k = 0
    class_origin = indices[0]
    print(int(class_origin))

    count_blocks = 0
    if pct >= 0:
        sum_blocks = 0
        breaking_pct_value = (pct / 100) * total_sum_blocks

    if verbose:
        print(f'total sum {total_sum_blocks}')
        print(f'breaking value {breaking_pct_value}')
    
    AOPC = []
    for i in range(len(blocks_dict.items())):
        ref_block_index = sorted_blocks[i][0]
        row = ref_block_index // block_per_row
        col = ref_block_index % block_per_row
        img_pred_aopc[row*block_size:row*block_size+block_size, col*block_size:col*block_size+block_size] =\
            perturbation(img_pred_aopc[row*block_size:row*block_size+block_size, col*block_size:col*block_size+block_size])
        
        input_tensor = preprocess_image(img_pred_aopc, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        probs, indices = torch.topk(torch.nn.Softmax(dim=-1)(output), k=1000)
        class_probs = dict(zip(indices.squeeze().tolist(), probs.squeeze().tolist()))
        
        # AOPC is a list of differences ----> f_c(x) at step 0 - f_c(x) at step i
        AOPC.append(probs_k0.item() - class_probs[int(class_origin)])

        count_blocks += 1
        if pct >= 0:
            sum_blocks += np.sum(sorted_blocks[i][1][0])
            if sum_blocks >= breaking_pct_value:
                if verbose:
                    print(f'breaking after {count_blocks} evaluated blocks')
                break

    # The dict is composed as following: {image_id: (original_class, AOPC, sum(AOPC)/L+1)}
    total_AOPC[name_id] = (class_origin.item(), AOPC, np.sum(AOPC)/count_blocks)


# Save results in .pkl format
with open(args.file_name, 'wb') as f:
    pickle.dump(total_AOPC, f)