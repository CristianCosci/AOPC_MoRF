import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image

# Directory path of images
directory_imgs = "imgs/"
# Resizing dimensions for image
width = 224
height = 224

# Create an empty list to memorize files names
file_list = []

# Go through each file in the folder
for filename in os.listdir(directory_imgs):
    # Check if the file is a .png image
    if filename.endswith(".png"):
        # Get the number "X" from the file name
        x = int(filename.split("_")[1].split(".")[0])
        # Save the file name and the number "X" in a list
        file_list.append((filename, x))

# Sort the list using the number "X" of each image
sorted_file_list = sorted(file_list, key=lambda x: x[1])

MODEL_gradcam = models.resnet50(pretrained=True)
TARGET_LAYERS = [MODEL_gradcam.layer4]

for name_id in range(len(sorted_file_list)):
    img = cv2.imread(directory_imgs+sorted_file_list[name_id][0], 1)[:, :, ::-1]
    resized_image = cv2.resize(img, (width, height))
    resized_image = np.float32(resized_image) / 255

    input_tensor = preprocess_image(resized_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cam_algorithm = EigenCAM

    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda=True) as cam:
        cam.batch_size = 256
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam_mask = grayscale_cam * 255  #make range between 0-255

    cv2.imwrite("cams/img_cam_"+str(name_id)+".png", grayscale_cam_mask)
