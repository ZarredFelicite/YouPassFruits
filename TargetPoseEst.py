# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math

import matplotlib.pyplot as plt
import PIL
from PIL import Image

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(box):
    #box = [x0, y0, x1, y1, conf, cls_pred]

    width = abs(box[0]-box[2])
    height = abs(box[1]-box[3])
    centerx = (box[0]+box[2])/2
    centery = (box[1]+box[3])/2
    box = [centerx, centery, int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most three types of targets in each image
    target_lst_box = [[], [], []]
    target_lst_pose = [[], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = person, 0 = not_a_target
    fname = base_dir / file_path
    img_vals = []
    f = open(fname, "r")
    temp = f.read().replace('[','').replace(']','').replace('\n', ' ').split(' ')
    
    #img_vals = np.array([[try(float(temp[j][i])), except ValueError: pass, for i in range(7)]for j in range(len(temp))])
     #float(i) for i in ... for j in ..
    removed_whitespace = [float(x) for x in temp if x !='']
    img_vals = []
    for num in range(len(removed_whitespace)):
        if num%6 == 0:
            curr = [removed_whitespace[num]]
        elif num%6 == 1:
            curr.append(removed_whitespace[num])
            img_vals.append(curr)
        else:
            curr.append(removed_whitespace[num])
    
    #with open(fname, "r") as f:
    #    for row in f.readlines():
    #        print(row)
    #        test = row.replace('[','').replace(']','').replace('\n', '').lstrip().split(' ')
    #        print(test)
    #        temp = [float(i) for i in test]
    #        img_vals.append(temp)
    #img_vals = f.read().replace('[','').replace(']','').replace('', '')#.split(' ')
    
    #set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    for target_num in img_vals:
        print(target_num)
        try:
            box = get_bounding_box(target_num) # [x,y,width,height]
            pose = image_poses[file_path] # [x, y, theta]
            target_lst_box[int(target_num[5])].append(box) # bounding box of target
            target_lst_pose[int(target_num[5])].append(np.array(pose).reshape(3,)) # robot pose
        except TypeError:
            pass

    # if there are more than one objects of the same type, combine them
    for i in range(3):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
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

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        print(target_num)
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]


        target_pose = {'y':0, 'x': 0}
        ######### Replace with your codes #########
        centx = box[0] + box[2]/2
        centy = box[1] + box[3]/2
        # depth = true_height*focal_length/(box[1] - box[3])
        depth = true_height*focal_length/(box[3])
        #print(box)
        #print(depth)
        # x = depth*(centx-320)/focal_length
        x = (depth * (box[0] - 320)) / focal_length
        print("ROBOT POSE", robot_pose)
        print(x)
        print(depth)
        dist = np.sqrt(depth**2 + x**2)
        # theta1 = math.atan(x[t]/depth[t])
        theta1 = np.arctan2(depth,x)
        # theta1 = np.arctan2(depth[t],x[t])
        # theta2 = robot_pose[2] - theta1
        targx = robot_pose[0] + dist*np.cos(robot_pose[2] + theta1 - np.pi/2)
        targy = robot_pose[1] + dist*np.sin(robot_pose[2] + theta1 - np.pi/2)
        # target_pose = {'y': np.round(targy,4), 'x': np.round(targx,4)}
        target_pose['x'] = targx[0]
        target_pose['y'] = targy[0]
        target_pose_dict[target_list[target_num-1]] = target_pose
        
        ###########################################
    print("targets:", target_pose_dict)
    return target_pose_dict

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, person_est = [], [], []
    target_est = {}
    
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('person'):
                person_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    new_apples = []
    appleset = set()
    print("_________________")
    print("APPLES: ", apple_est)
    for i in range(len(apple_est)):
        currx = apple_est[i][1]
        curry = apple_est[i][0]

        sum_x = 0
        sum_y = 0
        count = 0
        for k in range(len(apple_est)):
            difference = ((currx-apple_est[k][1])**2 + (curry-apple_est[k][0])**2)**(1/2)
            threshold = 0.2
            if difference < threshold and k not in appleset:
                appleset.add(k)
                sum_x +=apple_est[k][1]
                sum_y +=apple_est[k][0]
                count += 1
                #add to sum
        if count > 0:
            new_apples.append([sum_y/count, sum_x/count])
    apple_est = new_apples
                

    new_lemons = []
    lemonset = set()
    for i in range(len(lemon_est)):
        currx = lemon_est[i][1]
        curry = lemon_est[i][0]

        sum_x = 0
        sum_y = 0
        count = 0
        for k in range(len(lemon_est)):
            difference = ((currx-lemon_est[k][1])**2 + (curry-lemon_est[k][0])**2)**(1/2)
            threshold = 0.001
            if difference < threshold and k not in lemonset:
                lemonset.add(k)
                sum_x +=lemon_est[k][1]
                sum_y +=lemon_est[k][0]
                count += 1
                #add to sum
        if count > 0:
            new_lemons.append([sum_y/count, sum_x/count])
        print("_______________")
        print("lemon_est", lemon_est)
        print("new_lemon", new_lemons)
    lemon_est = new_lemons


    new_humans = []
    humanset = set()
    for i in range(len(person_est)):
        currx = person_est[i][1]
        curry = person_est[i][0]

        sum_x = 0
        sum_y = 0
        count = 0
        for k in range(len(person_est)):
            difference = ((currx-person_est[k][1])**2 + (curry-person_est[k][0])**2)**(1/2)
            threshold = 0.001
            if difference < threshold and k not in humanset:
                humanset.add(k)
                sum_x +=person_est[k][1]
                sum_y +=person_est[k][0]
                count += 1
                #add to sum
        if count > 0:
            new_humans.append([sum_y/count, sum_x/count])
        print("______________")
        print("person_est", person_est)
        print("new_person", new_humans)
        print("_______________")
    person_est = new_humans
    
    for i in range(3):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[i][0], 'x':apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[i][0], 'x':lemon_est[i][1]}
        except:
            pass
        try:
            target_est['person_'+str(i)] = {'y':person_est[i][0], 'x':person_est[i][1]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    #print(target_est)
    #print(target_map)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')



