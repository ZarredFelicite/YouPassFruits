import math
import numpy as np
from helper import *
import ast
from util.pibot import PenguinPi
from RRT_Class import *
from Obstacle import *
from rrt_star import *


# find lemons too close to person and apples too far from person using Euclidean distance (threshold = 0.5)######
def compute_dist(apple_list, lemon_list, person_list):
    apple_list = apple_list
    lemon_list = lemon_list
    person_list = person_list
    to_move = [] # an array containing tuples (person_num, fruit_num, fruit)
        #fruit --> apple = 0, lemon = 1

    #compute distance between all apples and all people
        #pick apple closest to each person
    #compute distance between all lemons and all people
    apple_dist = [[], [], []]
    i= 0
    for person in person_list:
        j = 0
        for apple in apple_list:
            apple_dist[i].append(np.linalg.norm(apple-person))
        for lemon in lemon_list:
            lemon_dist = np.linalg.norm(lemon-person)
            if lemon_dist < 0.5:
                to_move.append((i, j, 1))
            j=j+1
        i = i+1
    output = [0,1,2]
    minval = apple_dist[0][0]+apple_dist[1][1]+apple_dist[2][2]
    for i in range(3):
        for j in range(3):
            k = 3-(i+j)
            if j!=i:
                if apple_dist[0][i]+apple_dist[1][j]+apple_dist[2][k]<minval:
                    output = [i,j,k]
    
    for x in range(len(output)):
        if apple_dist[x][output[x]] > 0.5:
        #np.linalg.norm(apple_dist[output[x]])>0.5:
            to_move.append((x, output[x], 0))

    return to_move

###########################################################################

# Returns the next set of coordinates for the robot to move to
# outputs [x y] position and if next one is lemon or apple
# object_to_move, lemon_or_apple, person = next_object_to_move_to(list_of_objects_to_move, robot_pose, apple_gt, lemon_gt)
#def next_object_to_move_to(object_list, robot_position, apple_gt, lemon_gt):
#    object_list = object_list
#    robot_position = robot_position
#    lemon_or_apple = object_list[0][2]
#    min_distance = compute_distance_between_points(robot_position, object_list[0][1])
#    #position =
#    # Compute distance between robot and object
#   for i in range( len(object_list)):
#
#        p1 = robot_position
#        if lemon_or_apple==1:
#            p2 = lemon_gt[object_list[i][1]]
#        else:
#            p2 = apple_gt[object_list[i][2]]
#        distance = compute_distance_between_points(p1, p2)
#        if distance <= min_distance:
#            lemon_or_apple = object_list[i][2]
#            person = object_list[i][0]
#            min_distance=distance
#            position = p2
#
#    return position, lemon_or_apple, person


def next_position(object_position, object_type, human_position):
    # INPUTS
    # object_position: x and y coordinates of object
    # object_type: 1 if lemon, 0 if apple
    # human_list: list of human points
    # Outputs waypoint target for robot to move to

    object_position = object_position
    object_type = object_type
    human_position = human_position
    waypoint = []

    line_between_human_and_object = compute_line_through_points(object_position, human_position)

    for apple in apple_list:
        if abs(np.linalg.norm(apple - person)) < 0.5:
            dont_move_apple.append(apple)
            break
        elif abs(np.linalg.norm(apple - person)) > 0.5 and apple not in dont_move_apple:
            to_move['person_' + str(i)]['apple_' + str(j)] = apple
            to_move['person_' + str(i)]['dist_' + str(j)] = abs(np.linalg.norm(apple - person))
            j = j + 1

    if object_type == 1: #lemon
        waypoint = 1
    elif object_type == 0: #apple
        waypoint = 1

    if point_in_line(line_between_human_and_object, waypoint):
        return waypoint


    return waypoint

def find_points(human, fruit, fruit_type):
    multiplier = [0.4, 0.6]
    adder = [0.24, -0.2]
    [x,y] = fruit.ppi_view()
    [hx, hy] = human.ppi_view()

    current_distance = compute_distance_between_two_points(fruit.ppi_view(), human.ppi_view())

    newx = (x - hx)*(current_distance+adder[fruit_type])/current_distance + hx
    newy = (y - hy)*(current_distance+adder[fruit_type])/current_distance + hy
    
    waypoint = [newx, newy]
    
    newx1 = (x - hx)*(multiplier[fruit_type])/current_distance + hx
    newy1 = (y - hy)*(multiplier[fruit_type])/current_distance + hy
    
    movement_to_waypoint = [newx1, newy1]

    point2 = fruit.ppi_view()
    return waypoint, movement_to_waypoint, point2


 

def planPath(next_object_to_move, robot_pose, mapname):
    #write function to plan the path to be called from operate.py
    apple_gt, lemon_gt, person_gt, _,_ = parse_map(mapname)
    obstacle_list = create_obstacle_list(mapname)
    #inputs: the map (to get the objects)
    #output: the set of waypoints to move to an object and move it
    #object_to_move, lemon_or_apple, person = next_object_to_move_to(list_of_objects_to_move, robot_pose, apple_gt, lemon_gt)
    #next object to move is  tuple (person_num, fruit_num, fruit)
    #x = object_to_move[0]
    #y = object_to_move[1]
    # line generated between human and object
    human_coord = Block(person_gt[next_object_to_move[0]], 3) #[person_gt[next_object_to_move[0]][1], person_gt[next_object_to_move[0]][0]]
    #object_position = [x, y]
    
    if next_object_to_move[2] == 1: #it is a lemon
        fruit_coord = Block(lemon_gt[next_object_to_move[1]], 2)
        waypoint, movement_to_waypoint, point2 = find_points(human_coord, fruit_coord, next_object_to_move[2])
        
        
    else:
        fruit_coord = Block(apple_gt[next_object_to_move[1]], 1)
        waypoint, movement_to_waypoint, point2 = find_points(human_coord, fruit_coord, next_object_to_move[2])


    # generates path to waypoint using RRT
    # produces matrix of 2xn with coordinates of points for robot to traverse through
    start_pose = (-robot_pose[0][0]+1.5, -robot_pose[1][0]+1.5) #might need to change
    #waypoint = waypoint+[1.5,1.5]
    #print(waypoint)
    final_waypoint = (-waypoint[0] + 1.5, -waypoint[1]+1.5)
    print('final waypoint')
    print(final_waypoint)

    #systemRRT = RRT(start = start_pose, goal = final_waypoint, width = 3, height = 3,
    #                           obstacle_list = obstacle_list, expand_dis = 0.5, path_resolution = 0.1)

    #systemRRT = RRT_star(startpos = start_pose, endpos = final_waypoint, obstacles = obstacle_list, n_iter = 200, radius = 0.1, stepSize = 0.1)
    
    path_to_waypoint = None #systemRRT.planning(animation=False) #None
    #np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    
    pr = 0.1
    count = 0
    while path_to_waypoint == None:
        #systemRRT = RRT(start=start_pose, goal=final_waypoint, width=3, height=3,
        #                obstacle_list=obstacle_list, expand_dis=0.5, path_resolution=pr)
        #pr = pr/10
        path_to_waypoint = pathSearch(startpos = start_pose, endpos = final_waypoint, obstacles = obstacle_list, n_iter = 200, radius = 0.1, stepSize = 0.25)
        #systemRRT.planning()
        #generating a path 4 times to try to ensure the shortest path
        if path_to_waypoint is not None:
           cur_len = len(path_to_waypoint)
           for _ in range(7):
               new_path = pathSearch(startpos = start_pose, endpos = final_waypoint, obstacles = obstacle_list, n_iter = 200, radius = 0.1, stepSize = 0.25)
               if new_path is not None and len(new_path) < cur_len:
                   path_to_waypoint = new_path
                   cur_len = len(new_path)

        count = count + 1
        if count == 2:
            print("Unable to find path, moving to next object")

            #return 0
            path_to_waypoint = []

    if len(path_to_waypoint) > 0:
        final_path_to_waypoint = [[-path_to_waypoint[a][0]+1.5, -path_to_waypoint[a][1]+1.5] for a in range(len(path_to_waypoint)-2)]
    else:
        final_path_to_waypoint = []
    final_path_to_waypoint.append(waypoint)
    final_path_to_waypoint.append(point2) #move the object
    final_path_to_waypoint.append(movement_to_waypoint)
    final_path_to_waypoint.append(movement_to_waypoint)
    print("final path  ", final_path_to_waypoint)

    return final_path_to_waypoint

if __name__ == "__main__":
    next = (1, 0, 1)
    final = planPath(next, ([0],[0]), "map1.txt")
    print(final)
'''



STEPS
1. get map with positions of obstacles and current robot pose --> from map
2. generate list of apples and lemons to move away from humans --> to_move
3. find which object to be moved is the closest --> next_object_to_move_to
4. generate a line between the object to be moved and the human --> 
5. use line to find point on it for robot to move to
6. generate path to that point for robot to follow
7. robot moves to point on line  X
8. robot calculates how far to move object (?)
9. robot pushes object away or towards human
10. map and positions are updated and current robot pose is retrieved
11. step 3


'''

