# teleoperate the robot, perform SLAM and object detection

# basic python packages
from turtle import forward
import numpy as np
import cv2 
import os, sys
import time
import PySimpleGUI as sg
import os.path
from PIL import Image, ImageDraw
import ast
import base64
import io
import matplotlib.pyplot as plt
from torch import _euclidean_dist
import re
# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector
from pathPlanning import *
from helper import *
from mapcreator import *
from pathlib import Path

def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt, aruco_gt, aruco_indx = [], [], [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('aruco'):
                aruco_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                aruco_indx.append(int(key[-3]))
    #CHECK IF BELOW LINES ARE NEEDED
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_gt = person_gt[0:3]
    if len(aruco_gt) > 10:
        aruco_gt = aruco_gt[0:10]
    
    return apple_gt, lemon_gt, person_gt, aruco_gt, aruco_indx

def im_convert(img):
    #Convert original pibot image format to one that can be displayed using PySimpleGUI
    img = Image.fromarray(img.astype(np.uint8))
    with io.BytesIO() as output:
        img.save(output, format='PNG')
        data = output.getvalue()
    img_64 = base64.b64encode(data)
    return img_64

class Operate:
    def __init__(self, args):
        self.pibot = PenguinPi(args.ip, args.port)
        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip, pull_map=False)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'ticks': [20,5],
                        'autopathplan': False,
                        'stopplan': False,
                        'showmap': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.last_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        #self.detector_output = np.zeros([240,320], dtype=np.uint8)
        self.detector_output = np.zeros([1,6])
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.drive_flag = 0
        self.reached_goal_flag = 0
        self.target_list = ['apple', 'lemon', 'person']
        self.seen_objects = []
        self.seen_thresh = 0.3
        self.target_pose_dict = []

        #definitions for fully autonomous
        self.step = 0
        self.current_path = 0
        self.path_no = 0
        self.new_obstacle = 0
        self.to_move = []
        self.waypoint = []
        self.reverse = 0
        self.drawings = []
    def init_markers(self):
        apple_gt, lemon_gt, person_gt, aruco_gt, aruco_indx = parse_map('map1.txt')
        # self.ekf.taglist = aruco_indx

        self.ekf.taglist = aruco_indx
        self.ekf.markers = aruco_gt
        #print(self.ekf.P)
        for i in self.ekf.taglist:
            self.ekf.P = np.concatenate((self.ekf.P, np.zeros((2, self.ekf.P.shape[1]))), axis=0)
            self.ekf.P = np.concatenate((self.ekf.P, np.zeros((self.ekf.P.shape[0], 2))), axis=1)
            self.ekf.P[-2,-2] = 1e-10**2
            self.ekf.P[-1,-1] = 1e-10**2

            
    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'],
                self.command['ticks'][0],
                self.command['ticks'][1])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            if self.command['inference'] == True:
                self.detect_target()
                self.estimate_pose()
                #th = self.ekf.robot.state[2]
                #robot_xy = self.ekf.robot.state[0:2,:]
                #R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
                for object in self.target_pose_dict:
                    #lm_bff = np.array([object['xx'],object['depth']]).reshape(2,1)
                    #lm_inertial = robot_xy + R_theta @ lm_bff
                    #print(object['x']," ",object['y'])
                    lms.append(measure.Marker(np.array([object['depth'],-object['xx']]).reshape(2,1), object['id']))
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            # print('**********************')
            # for i in range(len(lms)):
            #    print('Position: {}  Tag: {}  '.format(lms[i].position,lms[i].tag))
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            #img320 = np.array(Image.fromarray(self.img).resize((320,240)))
            #Faster but less precise image downsampling
            img320 = self.img.reshape((240,2,320,2,3)).max(3).max(1)

            self.detector_output, self.network_vis = self.detector.detect_single_image(img320)

            self.command['inference'] = True
            #set to True for continuous inference testing
            self.file_output = (self.network_vis, self.ekf, self.detector_output)
            self.notification = f'{len(self.detector_output)} target type(s) detected'

    def get_bounding_box(self, box):
        #box = [x0, y0, x1, y1, conf, cls_pred]
        width = abs(box[0]-box[2])
        height = abs(box[1]-box[3])
        centerx = (box[0]+box[2])/2
        centery = (box[1]+box[3])/2
        box = [centerx, centery, int(width), int(height)] # box=[x,y,width,height]
        return box

    def object_id(self, label, x, y):

        if len(self.seen_objects)==0:
            id = 11
            self.seen_objects.append({'id': id, 'class':label, 'x': x, 'y': y})
        else:
            object_count = [0,0,0]
            new = True
            for seen in self.seen_objects:
                object_count[seen['class']]+=1
                if (seen['class']==label):
                    dist = ((seen['x'] - x)**2 + (seen['y']-y)**2)**0.5
                    if (dist<self.seen_thresh):
                        #seen object
                        new = False
                        #seen['x'] = (seen['x']+x)/2
                        #seen['y'] = (seen['y']+y)/2
                        id = seen['id']
            #print(object_count)
            if new:
                if object_count[label]>2:
                    print("discarding extra object: {}".format(label))
                    id = 0
                elif label==0:
                    id = 11+object_count[0]
                    object_count[0]+=1
                    print("Found object {} id: {}".format(label,id))
                    self.seen_objects.append({'id': id, 'class':label, 'x': x, 'y': y})
                elif label==1:
                    id = 14+object_count[1]
                    object_count[1]+=1
                    print("Found object {} id: {}".format(label,id))
                    self.seen_objects.append({'id': id, 'class':label, 'x': x, 'y': y})
                elif label==2:
                    id = 17+object_count[2]
                    object_count[2]+=1
                    print("Found object {} id: {}".format(label,id))
                    self.seen_objects.append({'id': id, 'class':label, 'x': x, 'y': y})
        if len(self.seen_objects)==9:
            self.seen_thresh = 0.5
        return id

    def estimate_pose(self):
        camera_matrix = self.ekf.robot.camera_matrix
        focal_length = camera_matrix[0][0]
        robot_pose = self.ekf.robot.state    #[[x], [y], [theta]]
        detections = self.detector_output
        # actual sizes of targets
        target_dimensions = []
        apple_dimensions = [0.075448, 0.074871, 0.071889]
        target_dimensions.append(apple_dimensions)
        #lemon_dimensions = [0.060588, 0.059299, 0.053017]
        lemon_dimensions = [0.060588, 0.059299, 0.0555]
        target_dimensions.append(lemon_dimensions)
        person_dimensions = [0.07112, 0.18796, 0.37592]
        target_dimensions.append(person_dimensions)
        target_list = self.target_list
        self.target_pose_dict = []
        for target_indx in range(len(detections)):
            detection = detections[target_indx] # [x0, y0, x1, y1, conf, cls_pred]
            target = int(detection[5])
            detection[0] = detection[0]/320*640
            detection[2] = detection[2]/320*640
            detection[1] = detection[1]/240*480
            detection[3] = detection[3]/240*480
            box = self.get_bounding_box(detection) # box=[x,y,width,height]
            true_height = target_dimensions[target][2]
            depth = true_height*focal_length/(box[3])
            x = (depth * (box[0] - 320)) / focal_length
            dist = np.sqrt(depth**2 + x**2)
            theta1 = np.arctan2(depth,x)
            targx = robot_pose[0] + dist*np.cos(robot_pose[2] + theta1 - np.pi/2)
            targy = robot_pose[1] + dist*np.sin(robot_pose[2] + theta1 - np.pi/2)
            id = self.object_id(target,targx[0],targy[0])
            if id:
                target_pose = {'id': id, 'class': target, 'x': targx[0], 'y': targy[0], 'xx':x, 'depth':depth}
                self.target_pose_dict.append(target_pose)

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip, pull_map):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)

        return EKF(robot, pull_map)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            base_dir = Path('./')
            makeamap(base_dir)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1], self.file_output[2])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas, window):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        #########################
        #ekf_data = pygame.image.tostring(ekf_view, 'RGB')
        #ekf_data = base64.b64encode(ekf_data)
        #robot_view = cv2.resize(self.aruco_img, (320, 240))
        #self.draw_pygame_window(canvas, robot_view, 
        #                        position=(h_pad, v_pad)
        #                        )
        # for target detector (M3)
        # detector_view = cv2.resize(self.network_vis,
        #                            (320, 240), cv2.INTER_NEAREST)
        #################################
        detector_view = self.network_vis
        window["-IMAGE-"].update(data=im_convert(detector_view))
        # self.draw_pygame_window(canvas, detector_view, 
        #                         position=(h_pad, 240+2*v_pad)
        #                         )

        # canvas.blit(self.gui_mask, (0, 0))
        # self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        # self.put_caption(canvas, caption='Detector',
        #                  position=(h_pad, 240+2*v_pad))
        # self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        # notifiation = TEXT_FONT.render(self.notification,
        #                                   False, text_colour)
        # canvas.blit(notifiation, (h_pad+10, 596))

        # time_remain = self.count_down - time.time() + self.start_time
        # if time_remain > 0:
        #     time_remain = f'Count Down: {time_remain:03.0f}s'
        # elif int(time_remain)%2 == 0:
        #     time_remain = "Time Is Up !!!"
        # else:
        #     time_remain = ""
        # count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        # canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    #run path planning
    def pathplanning(self):
        if self.command['autopathplan'] == True:
            mapname = "surveymap.txt"
            if self.to_move == []:
                apple_list, lemon_list, person_list, _,_ = parse_map(mapname)
                self.to_move = compute_dist(apple_list, lemon_list, person_list)
                self.new_obstacle = 1

            if self.new_obstacle == 1:
                print("Planning path")
                self.reverse = 1
                if self.current_path == 0:
                    robot_pose =  operate.ekf.robot.state
                    self.current_path = planPath(self.to_move[self.path_no], robot_pose, mapname)

                    if self.current_path == 0:
                        self.to_move.append(self.to_move[self.path_no])
                    self.path_no = self.path_no + 1
                if self.current_path != 0:
                    self.drive_flag = 1
                    self.waypoint = self.current_path[self.step]
                    self.showpath(self.current_path, window, g)
                    self.new_obstacle = 0
                    self.waypoint = self.current_path[self.step]
                    print(self.waypoint)
        return
           
    
    def followpath(self):
        if self.command['autopathplan'] == True and self.current_path !=0:
            if (self.step < len(self.current_path)-1):
                    robot_position = np.array([self.ekf.robot.state[0], self.ekf.robot.state[1]])
                    d1 = compute_distance_between_two_points(robot_position, self.current_path[self.step])

                    if (d1 < 0.025):  # Not sure about this threshold value
                        self.step += 1
                        self.waypoint = self.current_path[self.step]
                        print(self.waypoint)

            elif self.step == len(self.current_path)-1:
                if self.path_no < len(self.to_move):
                    print("Getting new path to: ", self.to_move[self.path_no])
                    make_obst_map(base_dir = Path('./'), remove = self.to_move[self.path_no-1])
                    self.new_obstacle = 1
                    self.step = 0
                    self.drive_flag = 0
                    self.current_path = 0
                    self.command['motion'] = [0,0]
                else:
                    self.command['stopplan'] = True
                    print("Finished route")
                    self.waypoint = []
                    self.to_move = []
                    self.drive_flag = 0
                    self.command['motion'] = [0,0]

        return 

    def showpath(self, current_path, window, g):
        path = [[] for _ in range(len(current_path))]
        for i in range(len(current_path)):
            path[i] = [(x+1.5)*(500/3) for x in current_path[i]]
            #g.DrawCircle(path[i], 2, fill_color='black')
        for i in range(len(path)-1):
            print("1: {} 2: {}".format(path[i],path[i+1]))
            g.DrawLine(path[i],path[i+1])
        print(path)
    def stopplanning(self, new_path):
        if self.command['stopplan'] == True:
            print("Stop planning")
            self.command['autopathplan'] = False
            self.command['stopplan'] = False
            new_path = 1
            self.drive_flag = 0
        return new_path


    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [1, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-1, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 1]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -1]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run pathPlanning
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.command['autopathplan'] = True
            #stop path planning
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                self.command['stopplan'] = True
             
            #produce map on GUI
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.command['showmap'] = True

            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                        # self.init_markers()
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.drive_flag = 1
        if self.quit:
            pygame.quit()
            sys.exit()

    def drive_to_point(self, waypoint, robot_pose, step, current_path):
        if self.drive_flag:
            robot_pose = robot_pose
            
            THRESHOLD = 0.005
            ALPHA_THRESHOLD = 0.07

            forward_vel = self.command['ticks'][0]
            turn_vel = self.command['ticks'][1]
            P_t = 20
            P_f = 30
            max_vel = 40
            min_vel = 25
            err_dist = ((robot_pose[0] - waypoint[0])**2 + (robot_pose[1] - waypoint[1])**2)**0.5 #euclidean distance
            
            if abs(robot_pose[0] - waypoint[0]) > THRESHOLD or abs(robot_pose[1] - waypoint[1]) > THRESHOLD:
                alpha, angle = get_angle_robot_to_goal(robot_pose, waypoint)
                err_ang = (abs(alpha)).item()
                if alpha > ALPHA_THRESHOLD:
                    v = 0
                    w = 1
                elif alpha < -ALPHA_THRESHOLD:
                    v = 0
                    w = -1
                else:
                    v = 1
                    w = 0
                forward_vel = max(min(int(P_f*err_dist),max_vel),min_vel)
                if (turn_vel>0.15)and(turn_vel<0.3):
                    turn_vel = 2
                elif (turn_vel>0.5)and(turn_vel<2):
                    turn_vel = 3
                else:
                    turn_vel = min(int(round(err_ang*P_t)),7)
                self.command['motion'] = [v, w]
                self.command['ticks'] = [forward_vel, turn_vel]
                self.reached_goal_flag = 0
            else:
                if self.command['autopathplan'] != True:
                    self.command['motion'] = [0,0]
                    self.reached_goal_flag = 1
                    self.drive_flag = 0
                    print("Reached Waypoint!")
                # lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
                # if (len(lms) < 2)and(step==len(current_path)-4):
                #     v = 0
                #     w = 1
                #     self.command['motion'] = [v, w]
                #     self.command['ticks'] = [0,6]
                #     print("in correction loop")
                # else:
                    
        #return waypoint
    def check_waypoint(self):
        if self.command['autopathplan'] == True and self.current_path !=0:
            L = len(self.current_path) - self.step
            if (L==1)and(self.reverse==1):
                print("STAARTING TIME METH")
                start_t = time.time()
                self.command['motion'] = [0, 0]
                self.take_pic()
                drive_meas = self.control()
                self.update_slam(drive_meas)
                self.record_data()
                self.save_image()
                while True:
                    print("reverse")
                    current_t = time.time()
                    t = current_t - start_t
                    self.command['motion'] = [-1, 0]

                    self.take_pic()
                    drive_meas = self.control()
                    self.update_slam(drive_meas)
                    self.record_data()
                    self.save_image()
                    if t > 1:
                        self.reverse = 0
                        self.command['motion'] = [0, 0]
                        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
                        while (len(lms) < 2):
                            v = 0
                            w = 1
                            self.command['motion'] = [v, w]
                            self.command['ticks'] = [0,10]
                            print("in correction loop")
                            self.take_pic()
                            lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
                            drive_meas = self.control()
                            self.update_slam(drive_meas)
                            self.record_data()
                            self.save_image()
                        self.command['motion'] = [0, 0]
                        break
        #             if t > 0.5:
        #                 start_t = time.time()
        #                 while True:
        #                     print("rotate left")
        #                     current_t = time.time()
        #                     t = current_t - start_t
        #                     operate.command['motion'] = [0, -1]
        #                     operate.take_pic()
        #                     drive_meas = operate.control()
        #                     operate.update_slam(drive_meas)
        #                     operate.record_data()
        #                     operate.save_image()

        #                     if t > 5:
        #                         start_t = time.time()
        #                         while True:
        #                             print("rotate right")
        #                             current_t = time.time()
        #                             t = current_t - start_t
        #                             operate.command['motion'] = [0, 1]
        #                             operate.take_pic()
        #                             drive_meas = operate.control()
        #                             operate.update_slam(drive_meas)
        #                             operate.record_data()
        #                             operate.save_image()

        #                             if t > 5:
        #                                 operate.command['motion'] = [0, 0]
        #                                 operate.take_pic()
        #                                 drive_meas = operate.control()
        #                                 operate.update_slam(drive_meas)
        #                                 operate.record_data()
        #                                 operate.save_image()
        #                                 break
        #                         break
        #                 break

        return

    def showmap(self, window, g):
        if self.command['showmap'] == True:
            if self.drawings:
                for item in self.drawings:
                    g.DeleteFigure(item)
            else:
                self.drawings = []
            print("showing map")
            apple_gt, lemon_gt, person_gt, aruco_gt, aruco_indx = parse_map("surveymap.txt")#surveymap.txt")
            apple = [list(x)[::-1] for x in apple_gt]
            lemon = [list(x)[::-1] for x in lemon_gt]
            person = [list(x)[::-1] for x in person_gt]
            aruco = [list(x)[::-1] for x in aruco_gt]
            icon_size = 7
            radius = 83.5
            scale = g_size/3
            l_vel, r_vel = 0,0
            for i in range(len(apple)):
                apple[i] = [(x+1.5)*scale for x in apple[i]]
            for i in range(len(lemon)):
                lemon[i] = [(x+1.5)*scale for x in lemon[i]]
            for i in range(len(person)):
                person[i] = [(x+1.5)*scale for x in person[i]]
            for i in range(len(person)):
                self.drawings.append(g.DrawCircle(person[i], radius, fill_color='grey'))
            for i in range(len(apple)):
                self.drawings.append(g.DrawCircle(apple[i], icon_size, fill_color='red'))
            for i in range(len(lemon)):
                self.drawings.append(g.DrawCircle(lemon[i], icon_size, fill_color='yellow'))
            for i in range(len(person)):
                self.drawings.append(g.DrawCircle(person[i], icon_size, fill_color='blue'))
            for i in range(len(aruco)):
                aruco[i] = [(x+1.5)*scale for x in aruco[i]]
                self.drawings.append(g.DrawCircle(aruco[i], icon_size, fill_color='white'))
            self.command['showmap'] = False
        return
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    operate = Operate(args)
    timeout = False
    waypoint = [0,0]

    sg.theme("Python")
    g_size = 500
    pibot = PenguinPi(ip='localhost', port=40000)
    left_col = [
        [sg.Text("Robot View")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Clear Waypoint")],
        [sg.Button("Toggle Object Detection")],
        #[sg.Button("Fix Marker Covariance")],
        [sg.Button("Speed Boost")],
        [sg.Button("Exit")]
    ]
    middle_col = [
        [sg.Text("Map")],
        [sg.Graph((g_size, g_size), (g_size, g_size), (0, 0),
                  key='-GRAPH-',
                  change_submits=True,
                  background_color='grey',
                  drag_submits=False,
                  enable_events=True)]
    ]
    # right_col = [
    #     [sg.Text("SLAM Map View")],
    #     [sg.Image(filename="", key="-IMAGE_EKF-")],
    # ]
    layout = [
        [
            sg.Column(left_col),
            sg.VSeperator(),
            sg.Column(middle_col),
            #sg.VSeperator(),
            #sg.Column(right_col),
        ]
    ]

    window = sg.Window('Robot Controller', layout, return_keyboard_events=True, finalize=True, resizable=True)
    g = window['-GRAPH-']
    icon_size = 7
    radius = 83.5

    pointer = g.DrawPoint((0,0), icon_size, color='black')
    bot_loc = g.DrawPoint((0,0), icon_size, color='black')
    waypoint = (0,0)
    inference = True

    while True:
        # print(operate.ekf.robot.state)
        operate.showmap(window,g)
        operate.update_keyboard()
        
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        operate.pathplanning()
        operate.followpath()
        operate.check_waypoint()
        g.DeleteFigure(bot_loc)
        bot_loc = [operate.ekf.robot.state[0,0]+1.5,operate.ekf.robot.state[1,0]+1.5]
        bot_loc = [x*(g_size/3) for x in bot_loc]
        bot_loc = g.DrawCircle(bot_loc, 3, fill_color='green')
        waypoint = operate.waypoint
        operate.drive_to_point(waypoint, operate.ekf.robot.state, operate.step, operate.current_path)

        

###################################################################################
            #waypoint = current_path[step]
            #robot_position = np.array([operate.ekf.robot.state[0], operate.ekf.robot.state[1]])
            #d1 = compute_distance_between_two_points(robot_position, waypoint)
            #print("distance: ", d1)
            #print("tomove: ", to_move)
            #print("path_no", path_no)
            #print("numer to move", )
            #if (len_path > step):
            #    if (d1 < 0.1):
            #        step = step + 1


            #print("robotposition : ", operate.ekf.robot.state,)
            #print("waypoint : ", waypoint)
 ######################################################################
        # visualise
        operate.draw(canvas, window)
        pygame.display.update()
        
        event, values = window.read(timeout=10)
        mouse = values['-GRAPH-']
        #Manual movement
        if event == 'Up:111':
            operate.command['motion'] = [1,0]
        elif event == 'Down:116':
            operate.command['motion'] = [-1,0]
        elif event == "Right:114":
            operate.command['motion'] = [0,-1]
        elif event == 'Left:113':
            operate.command['motion'] = [0,1]
        if event == '-GRAPH-':
            if mouse == (None, None):
                continue
            operate.waypoint = [mouse[0]/g_size*3-1.5, mouse[1]/g_size*3-1.5]
            print(operate.waypoint)
            operate.drive_flag = 1
            g.DeleteFigure(pointer)
            pointer = g.DrawPoint(mouse, icon_size, color='black')
        # if event in ('Fix Marker Covariance'):
        #     for i in range(1,len(operate.ekf.P)-3):
        #         operate.ekf.P[-i,:] = 1e-7
        #         operate.ekf.P[:,-i] = 1e-7
        #     print("settings covariance to 0 for markers")
        if event in ('Speed Boost'):
            operate.command['ticks'] = [operate.command['ticks'][0] + 5, operate.command['ticks'][1] + 2]
            #operate.command['ticks'][0] = operate.command['ticks'][0] + 5
            print('Speed forward: {} turn: {}'.format(operate.command['ticks'][0],operate.command['ticks'][1]))

        if event in ('Toggle Object Detection'):
            inference = not inference
            print("Inference is set to: {}".format(inference))
        if inference:
            operate.command['inference'] = True
        else:
            operate.command['inference'] = False
        if event in ('Clear Waypoint'):
            operate.drive_flag = 0
            print("Clearing Waypoint")
            operate.command['motion'] = [0, 0]
        if event in (sg.WIN_CLOSED, 'Exit'):
            operate.command['motion'] = [0, 0]
            time.sleep(1)
            break
    window.close()

