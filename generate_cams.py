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

# definisci il percorso della cartella contenente le immagini
directory_imgs = "ssim_eigencam_target_imgs_5f/"
# ridimensiona l'immagine
width = 224
height = 224

# crea una lista vuota per memorizzare i nomi dei file
file_list = []

# scorri tutti i file nella cartella
for filename in os.listdir(directory_imgs):
    # controlla se il file è un'immagine JPEG
    if filename.endswith(".png"):
        # estrai il numero "X" dal nome del file
        x = int(filename.split("_")[1][3:])
        # aggiungi il nome del file e il numero "X" alla lista
        file_list.append((filename, x))

# ordina la lista in base al numero "X" presente nel nome del file
sorted_file_list = sorted(file_list, key=lambda x: x[1])

MODEL_gradcam = models.resnet50(pretrained=True)
TARGET_LAYERS = [MODEL_gradcam.layer4]

for name_id in range(len(sorted_file_list)):
    img = cv2.imread(directory_imgs+sorted_file_list[name_id][0], 1)[:, :, ::-1]
    resized_image = cv2.resize(img, (width, height))
    #cv2.imwrite("center_2_img/best_img"+str(name_id)+"_.png", resized_image[:,:,::-1])

    resized_image = np.float32(resized_image) / 255

    input_tensor = preprocess_image(resized_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    cam_algorithm = EigenCAM

    with cam_algorithm(model = MODEL_gradcam, target_layers = TARGET_LAYERS, use_cuda=True) as cam:
        cam.batch_size = 256
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam_mask = grayscale_cam * 255  #make range between 0-255

    cv2.imwrite("ssim_eigencam_target_cams_5f/best_img_cam"+str(name_id)+"_.png", grayscale_cam_mask)


