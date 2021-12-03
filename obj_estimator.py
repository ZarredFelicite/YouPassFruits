import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math

import matplotlib.pyplot as plt
import PIL

import os 
import time

import numpy as np

import cv2
from PIL import Image, ImageDraw
from pytorchyolo import detect, models

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

model = models.load_model(
  "network/scripts/PyTorch-YOLOv3/config/yolov3-tiny-custom.cfg", 
  "network/scripts/PyTorch-YOLOv3/weights/yolov3_tiny_v5.weights")
print(model)
classes = ["Apple","Lemon", "Human"]
img_size = 640
aheight = 0.072 #x=0.075 #y=0.076 #z=0.072
lheight = 0.053 #x=0.059 #y=0.061 #z=0.053
hheight = 0.37592
image_width = 640
image_height = 480
focallength = 5.922383032090032202e+02 #or 5.823218803994668633e+02

dict = {}
num_lines = sum(1 for line in open('lab_output/images.txt'))
with open("lab_output/images.txt") as f_images:
    for i in range(num_lines):
        dict['image_'+str(i)] = ast.literal_eval(f_images.readline())
        image = np.array(Image.open("lab_output/pred_" + str(i)+".png"))
        prediction = detect.detect_image(model, image, img_size=img_size, conf_thres=0.1, nms_thres = 0.2)
        np.savetxt("lab_output/pred_"+str(i)+".txt", prediction)

# fileK = "{}intrinsic.txt".format('./calibration/param/')
# camera_matrix = np.loadtxt(fileK, delimiter=',')
# focal_length = camera_matrix[0][0]
# # actual sizes of targets
# target_dimensions = []
# apple_dimensions = [0.075448, 0.074871, 0.071889]
# target_dimensions.append(apple_dimensions)
# lemon_dimensions = [0.060588, 0.059299, 0.053017]
# target_dimensions.append(lemon_dimensions)
# person_dimensions = [0.07112, 0.18796, 0.37592]
# target_dimensions.append(person_dimensions)
# target_list = ['apple', 'lemon', 'person']
# num_lines = sum(1 for line in open("detections.txt"))
# detections_loc = np.zeros((num_lines,9))
# with open("detections.txt") as detections:
#     for j in range(num_lines):
#         box = list(filter(None,detections.readline().replace("\n","").split(" ")))
#         box = [float(i) for i in box]
#         x = box[0] + box[2]/2
#         y = box[1] + box[3]/2
#         width = box[2]-box[0]
#         height = box[3]-box[1]
#         conf = box[4]
#         obj = box[5]
#         image = box[6]
#         true_height = target_dimensions[int(obj)][2]
#         robot_pose = dict["image_"+str(int(image))]["pose"]

#         depth = true_height*focal_length/height
#         X = depth*(x-320)/focal_length
#         dist = math.sqrt(depth**2 + X**2)
#         theta1 = math.atan(X/depth)
#         theta2 = robot_pose[2][0] - theta1
#         targx = robot_pose[0][0] + dist*math.cos(theta2)
#         targy = robot_pose[1][0] + dist*math.sin(theta2)
#         #target_pose = {'y': np.round(targy,1), 'x': np.round(targx,1)}
#         detections_loc[j] = np.array([x,y,width,height,conf,obj,image,np.round(targy,5),np.round(targx,5)])
#         print(detections_loc[j].tolist())
# #Compressing multiple detections
# for j in range(detections_loc.shape[0]):
#     for i in range(detections_loc.shape[0]):
#         if ((detections_loc[j][5]==detections_loc[i][5])and(abs(detections_loc[j][7]-detections_loc[i][7])<thresh)and(abs(detections_loc[j][8]-detections_loc[i][8])<thresh)and(j!=i)):
#             #print("hit!")
#             #detections_loc[j]
#             pass