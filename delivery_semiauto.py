# semi-automatic approach for fruit delivery

# import modules
import sys, os
import ast
import numpy as np
import json
import argparse

sys.path.insert(0, "../util")
from util.pibot import PenguinPi

def get_angle_robot_to_goal(robot_state = np.zeros(3), goal = np.zeros(2)):
    x_goal, y_goal = goal
    x, y, theta = robot_state
    x_diff = x_goal - x
    y_diff = y_goal - y
    angle = np.arctan2(y_diff, x_diff)
    alpha = clamp_angle(angle - theta)
    return alpha, angle

def get_distance_robot_to_goal(robot_state = np.zeros(3), goal = np.zeros(2)):
    
    x_goal, y_goal= goal
    x, y,_ = robot_state
    x_diff = x_goal - x
    y_diff = y_goal - y

    rho = np.hypot(x_diff, y_diff)

    return rho

def clamp_angle(rad_angle = 0, min_value = -np.pi, max_value = np.pi):
    if min_value > 0:
        min_value *= -1
    angle = (rad_angle + max_value) % (2 * np.pi) + min_value

    return angle



# read in the object poses, note that the object pose array is [y,x]
def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt = [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
    
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_egt = person_gt[0:3]
    
    return apple_gt, lemon_gt, person_gt

# find lemons too close to person and apples too far from person using Euclidean distance (threshold = 0.5)
def compute_dist(apple_list, lemon_list, person_list):
    apple_list = apple_list
    lemon_list = lemon_list
    person_list = person_list
    to_move = {}
    i = 0
    for person in person_list:
        to_move['person_'+str(i)] = {}
        j,k = 0,0
        # find apples that are too far
        for apple in apple_list:
            if abs(np.linalg.norm(apple-person)) > 0.5:
                to_move['person_'+str(i)]['apple_'+str(j)] = apple
                to_move['person_'+str(i)]['dist_'+str(j)] = abs(np.linalg.norm(apple-person))
                j = j+1
            try: 
                to_move['person_'+str(i)]['apple_2']
                print('All apples too far from Person:', person)
            except:
                pass
        # find lemons that are too close
        for lemon in lemon_list:
            if abs(np.linalg.norm(lemon-person)) < 0.5:
                to_move['person_'+str(i)]['lemon_'+str(k)] = lemon
                to_move['person_'+str(i)]['dist_'+str(k)] = abs(np.linalg.norm(lemon-person))
                print('There are lemons too close to Person:', person)
            k = k+1
        i = i+1
    return to_move

# semi-automatic delivery approach by providing a series of waypoints to guide the robot to move all targets
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    waypoint = waypoint
    robot_pose = robot_pose
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    wheel_vel = 30 # tick
            # Need to find the time to do 1 rotation
            # Time to rotate = (theta)/(2*pi) * 1 rotation time
    ROTATION_TIME = 3.1 # NEED TO CHANGE THIS LATER ONCE CALIBRATION IS DONE
    TRAVEL_TIME = 0# time it takes to travel 1 m
    THRESHOLD = 0.01
    ALPHA_THRESHOLD = 0.01
    while abs(robot_pose[0] - waypoint[0]) > 0.01 or abs(robot_pose[1] - waypoint[1]) > 0.01:
        alpha, angle = get_angle_robot_to_goal(robot_pose, waypoint)
        if alpha > ALPHA_THRESHOLD:
            v = 0
            w = 1
            ppi.set_velocity([0,  1], turning_tick=wheel_vel)
        elif alpha < -ALPHA_THRESHOLD:
            v = 0
            w = -1
            ppi.set_velocity([0,  -1], turning_tick=wheel_vel)
        else:
            v = 1
            w = 0
            ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ppi.set_velocity([v, w], tick=wheel_vel, time=drive_time)
    operate.update_slam(drive_meas)



    # alpha, angle = get_angle_robot_to_goal(robot_pose, waypoint)
    
    # # turn towards the waypoint
    # turn_time = np.abs(alpha / (2*np.pi) * ROTATION_TIME) # replace with your calculation
    # print("Turning for {:.2f} seconds".format(turn_time))
    # print("rotational vel: ", np.sign(alpha) * 1)
    # if alpha > 0:
    #     ppi.set_velocity([0,  1], turning_tick=wheel_vel, time=turn_time)
    # else:
    #     ppi.set_velocity([0,  -1], turning_tick=wheel_vel, time=turn_time)
    
    # # after turning, drive straight to the waypoint

    # dist = get_distance_robot_to_goal(robot_pose, waypoint)

    # drive_time = dist * TRAVEL_TIME # replace with your calculation
    # print("Driving for {:.2f} seconds".format(drive_time))
    # ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    # # update the robot pose [x,y,theta]
    # robot_pose = [waypoint[0],waypoint[1],angle] # replace with your calculation
    ####################################################
    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit delivery")
    parser.add_argument("--map", type=str, default='map1.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the map
    apple_gt, lemon_gt, person_gt = parse_map(args.map)
    print("Map: apple = {}, lemon = {}, person = {}".format(apple_gt, lemon_gt, person_gt))

    # find apple(s) and lemon(s) that need to be moved
    to_move = compute_dist(apple_gt, lemon_gt, person_gt)
    print("Fruits to be moved: ", to_move)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    # semi-automatic approach for fruit delivery
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue
        
        # robot drives to the waypoint
        waypoint = [x,y]
        robot_pose = drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break