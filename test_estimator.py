import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math

import matplotlib.pyplot as plt
import PIL

thresh = 0.01
dict = {}
num_lines = sum(1 for line in open('lab_output/images.txt'))
f = open("detections.txt", "w")
with open("lab_output/images.txt") as f_images:
    for i in range(num_lines):
        dict['image_'+str(i)] = ast.literal_eval(f_images.readline())
        with open("lab_output/pred_"+str(i)+".txt") as f_detections:
            data = f_detections.read().replace("[","").replace("]","").replace("\n","").split(" ")
            #data = np.fromstring(data)
            data = list(filter(None, data))
            data = [float(i) for i in data]
            length = len(data)
            detections = np.append(np.resize(np.array(data),(int(length/6),6)), np.ones((int(length/6),1))*i, axis=1)
            for i in range(int(length/6)):
                string = np.array2string(detections[i], separator=" ").replace("[","").replace("]","").replace("\n","")
                f.write(string)
                f.write("\n")
f.close()
fileK = "{}intrinsic.txt".format('./calibration/param/')
camera_matrix = np.loadtxt(fileK, delimiter=',')
focal_length = camera_matrix[0][0]
# actual sizes of targets
target_dimensions = []
apple_dimensions = [0.075448, 0.074871, 0.071889]
target_dimensions.append(apple_dimensions)
lemon_dimensions = [0.060588, 0.059299, 0.053017]
target_dimensions.append(lemon_dimensions)
person_dimensions = [0.07112, 0.18796, 0.37592]
target_dimensions.append(person_dimensions)
target_list = ['apple', 'lemon', 'person']
num_lines = sum(1 for line in open("detections.txt"))
detections_loc = np.zeros((num_lines,9))
with open("detections.txt") as detections:
    for j in range(num_lines):
        box = list(filter(None,detections.readline().replace("\n","").split(" ")))
        box = [float(i) for i in box]
        x = box[0] + box[2]/2
        y = box[1] + box[3]/2
        width = box[2]-box[0]
        height = box[3]-box[1]
        conf = box[4]
        obj = box[5]
        image = box[6]
        true_height = target_dimensions[int(obj)][2]
        robot_pose = dict["image_"+str(int(image))]["pose"]

        depth = true_height*focal_length/height
        X = depth*(x-320)/focal_length
        dist = math.sqrt(depth**2 + X**2)
        theta1 = math.atan(X/depth)
        theta2 = robot_pose[2][0] - theta1
        targx = robot_pose[0][0] + dist*math.cos(theta2)
        targy = robot_pose[1][0] + dist*math.sin(theta2)
        #target_pose = {'y': np.round(targy,1), 'x': np.round(targx,1)}
        detections_loc[j] = np.array([x,y,width,height,conf,obj,image,np.round(targy,5),np.round(targx,5)])
        print(detections_loc[j].tolist())
#Compressing multiple detections
for j in range(detections_loc.shape[0]):
    for i in range(detections_loc.shape[0]):
        if ((detections_loc[j][5]==detections_loc[i][5])and(abs(detections_loc[j][7]-detections_loc[i][7])<thresh)and(abs(detections_loc[j][8]-detections_loc[i][8])<thresh)and(j!=i)):
            #print("hit!")
            #detections_loc[j]
            pass