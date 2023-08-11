# Importing libraries
import torch
import cv2
from tensorflow.keras.utils import load_img
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from flask import request, Blueprint
from util import load_model, make_transparent_foreground, remove_background

router = Blueprint("router", __name__)

@router.route("/")
def hello():
    return "This is the implementation for removing image background"

@router.route('/remove_background', methods=['POST'])
def remove_background_main():
    file = request.files.get('file')
    img_bytes = file.read()

    if os.path.isdir('inital_image') == False:
        os.mkdir('inital_image')
    img_path = "./inital_image/test.jpg"

    with open(img_path, "wb") as img:
        img.write(img_bytes)
    
    img = load_img(img_path, target_size=(224, 224, 3))   
    deeplab_model = load_model()
    foreground = remove_background(deeplab_model, img)
    print("\n Background removed \n")
    image_fg = Image.fromarray(foreground)

    if os.path.isdir('final_image') == False:
        os.mkdir('final_image')
    image_path = 'final_image/final_foreground_image.jpg'
    image_fg.save(image_path)
    
    return 'Background removal successful, final image path:  {}'.format(image_path)
