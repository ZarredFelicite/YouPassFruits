import os 
import time
import numpy as np

import cv2
from PIL import Image, ImageDraw
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        #Load model and weights
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='network/scripts/best.pt')#.eval()
        self.model.conf = 0.94 #0.93
        self.model.iou = 0.1
    def detect_single_image(self, np_img):
        detections = self.model(np_img)
        pred = detections.xyxy[0].detach().cpu().numpy()
        #format = x1,y1,x2,y2,conf,class
        deleting = []
        for i in range(pred.shape[0]):
            if pred[i] is not None:
                if (pred[i,4] < 0.96)and(pred[i,5]==2):
                    deleting.append(i)
                if (pred[i,0]<20)or(pred[i,2]>220):
                    deleting.append(i)
        pred = np.delete(pred, deleting, axis=0)
        #print(pred)
        colour_map = self.visualise_output(pred, np_img)
        return pred, colour_map

    def visualise_output(self, detections, image):
        img = Image.fromarray(image)
        for j in range(detections.shape[0]):
            if int(detections[j][5])<3:
                draw = ImageDraw.Draw(img)
                colours = ["red", "yellow", "blue"]
                draw.rectangle([detections[j][0],detections[j][1],detections[j][2],detections[j][3]], outline=colours[int(detections[j][5])], width=3)
        image_boxes = np.array(img)
        return image_boxes