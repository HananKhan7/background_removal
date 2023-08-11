# Importing libraries
import torch
import cv2
from torchvision import transforms
import numpy as np

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # only evaluation and not training because its pre-trained and requires no additional training, in this case.
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # merge the colors into array format
    img= cv2.merge([b, g, r], 3)
    # create a transparent background
    bg = np.zeros(img.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, img, bg).astype(np.uint8)
    return foreground

def remove_background(model, input_file):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    #Based on model compatibility
    ])

    input_tensor = preprocess(input_file)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)
    
    foreground = make_transparent_foreground(input_file ,bin_mask)

    return foreground