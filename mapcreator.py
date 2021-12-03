#Reading the Slam and img maps and generating the map for the robot to use for path generation


#called by operate.py
#read in slam.txt and targets.txt -- both in lab_output folder
#generate surveymap
#save in normal folder

import os
from pathlib import Path
import ast
import json
import numpy as np

def generate_targets(fname: str, obs_dict) -> dict:
    with open(fname,'r') as f:
        
        usr_dict = ast.literal_eval(f.read())
        slammap = usr_dict["map"]
        for (i, tag) in enumerate(usr_dict["taglist"]):
            if tag<11:
                pass
            elif tag<14:
                #apples 11, 12, 13
                num = tag-11
                key = "apple_" + str(num)
                obs_dict[key] = {'y':slammap[1][i], 'x':slammap[0][i]}
            elif tag<17:
                #lemons 14, 15, 16
                num = tag-14
                key = "lemon_" + str(num)
                obs_dict[key] = {'y':slammap[1][i], 'x':slammap[0][i]}
            elif tag<20:
                #people 17, 18, 19
                num = tag-17
                key = "person_" + str(num)
                obs_dict[key] = {'y':slammap[1][i], 'x':slammap[0][i]}
        
    return obs_dict #apple_gt, lemon_gt, person_gt,

def find_markers(fname: str, obs_dict) -> dict:
    with open(fname, 'r') as f:
        
        usr_dict = ast.literal_eval(f.read())
        slammap = usr_dict["map"]
        for (i, tag) in enumerate(usr_dict["taglist"]):
            if tag<11:
                key = "aruco"+str(tag)+"_0"
            elif tag<14:
                #apples 11, 12, 13
                num = tag-11
                key = "apple_" + str(num)
            elif tag<17:
                #lemons 14, 15, 16
                num = tag-14
                key = "lemon_" + str(num)
            elif tag<20:
                #people 17, 18, 19
                num = tag-17
                key = "person_" + str(num)
            obs_dict[key] = {'y':slammap[1][i], 'x':slammap[0][i]}
    
    return obs_dict

def parse_map_obs(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt, aruco_gt = [], [], [], []
        apple_ind, lemon_ind, person_ind, aruco_ind = [], [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                apple_ind.append(int(key[-1]))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                lemon_ind.append(int(key[-1]))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                person_ind.append(int(key[-1]))
            elif key.startswith('aruco'):
                aruco_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                aruco_ind.append(int(key[-3]))

    
    return apple_gt, lemon_gt, person_gt, aruco_gt, apple_ind, lemon_ind, person_ind, aruco_ind

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict



def makeamap(base_dir = Path('./')):
    obs_dict = {}
    only_obs = {}
    obs_dict = find_markers(base_dir/'lab_output/slam.txt', obs_dict)
    only_obs = generate_targets(base_dir/'lab_output/slam.txt', only_obs)

    # save object and marker estimations
    with open(base_dir/'surveymap.txt', 'w') as fo:
        json.dump(obs_dict, fo)
    
    with open(base_dir/'obstmap.txt', 'w') as fo:
        json.dump(obs_dict, fo)

    #saving objects into targets.txt
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(only_obs, fo)

    print('New Map Created!')
    return

def make_obst_map( base_dir = Path('./'), remove = ()):
    #remove = (person_num, fruit_num, fruit)
    #fruit --> apple = 0, lemon = 1
    if remove != ():
        apple_gt, lemon_gt, person_gt, aruco_gt, apple_ind, lemon_ind, person_ind, aruco_ind = parse_map_obs('obstmap.txt')
        obs_dict = {}
        print(aruco_ind)

        for p in range(len(person_gt)):
            key = "person_" + str(person_ind[p])
            obs_dict[key] = {'y':person_gt[p][0], 'x':person_gt[p][1]}
        for a in range(len(aruco_gt)):
            key = "aruco"+str(aruco_ind[a])+"_0"
            obs_dict[key] = {'y':aruco_gt[a][0], 'x':aruco_gt[a][1]}
        for l in range(len(lemon_gt)):
            if remove[2] == 1 and lemon_ind[l] == remove[1]:
                pass
            else:
                key = "lemon_" + str(person_ind[l])
                obs_dict[key] = {'y':person_gt[l][0], 'x':person_gt[l][1]}
        for a in range(len(apple_gt)):
            if remove[2] == 0 and apple_ind[a] == remove[1]:
                pass
            else:
                key = "apple_" + str(apple_ind[a])
                obs_dict[key] = {'y':apple_gt[a][0], 'x':apple_gt[a][1]}

        # save object and marker estimations
        with open(base_dir/'obstmap.txt', 'w') as fo:
            json.dump(obs_dict, fo)


        print('Obstacle removed!')
    return


if __name__ == "__main__":
    
    base_dir = Path('./')
    makeamap(base_dir)
    make_obst_map( base_dir = Path('./'), remove = (2,2,0))
    