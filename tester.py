from __future__ import division
import cv2
from pytorchyolo import detect, models

import os
import argparse
import tqdm
import random
import numpy as np
from mss import mss
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator



# Load the YOLO model
model = models.load_model(
  "C:/bin/m3/yolov3-tiny-custom.cfg", 
  "C:/bin/m3/yolov3_tiny_v5.weights")
  
classes = ["Apple","Lemon", "Human"]

vid = cv2.VideoCapture(1)
img_size = 640
bounding_box = {'top': 400, 'left': 400, 'width': img_size, 'height': img_size}
sct = mss()

while(True):
    if False:
      # Capture the video frame
      # by frame
      ret, frame = vid.read()
      # Display the resulting frame
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = cv2.resize(frame, (img_size,img_size))
    sct_img = sct.grab(bounding_box)
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    image = np.array(img)
    # Runs the YOLO model on the image
    detections = detect.detect_image(model, image, img_size=640, conf_thres=0.1)
    print(detections)
    #detections = torch.from_numpy(detections)
    #draw markers
    for j in range(detections.shape[0]):
        draw = ImageDraw.Draw(img)
        colours = ["red", "green", "blue"]
        draw.rectangle([detections[j][0],detections[j][1],detections[j][2],detections[j][3]], outline=colours[int(detections[j][5])], width=4)
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('screen', image)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()