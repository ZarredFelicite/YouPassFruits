import PySimpleGUI as sg
import os.path
from util.pibot import PenguinPi # access the robot
from PIL import Image, ImageDraw
import numpy as np
import cv2
import ast
import io
import matplotlib.pyplot as plt
import time

def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt , aruco_gt = [], [], [], []

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
    
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_gt = person_gt[0:3]
    if len(aruco_gt) > 10:
        aruco_gt = aruco_gt[0:10]
    
    return apple_gt, lemon_gt, person_gt, aruco_gt


def main():
    sg.theme("Python")
    g_size = 500
    pibot = PenguinPi(ip='localhost', port=40000)
    right_col = [
        [sg.Text("Robot View")],
        [sg.Image(filename="", key="-IMAGE-")],
        [sg.Button("Exit")]
    ]
    left_col = [
        [sg.Text("Map")],
        [sg.Graph((g_size, g_size), (0, g_size), (g_size, 0), 
                  key='-GRAPH-',
                  change_submits=True,
                  background_color='grey',
                  drag_submits=False,
                  enable_events=True)]
    ]
    layout = [
        [
            sg.Column(left_col),
            sg.VSeperator(),
            sg.Column(right_col),
        ]
    ]

    window = sg.Window('Robot Controller', layout, finalize=True, resizable=True)
    g = window['-GRAPH-']
    apple_gt, lemon_gt, person_gt, aruco_gt = parse_map("map1.txt")
    apple = [list(x) for x in apple_gt]
    lemon = [list(x) for x in lemon_gt]
    person = [list(x) for x in person_gt]
    aruco = [list(x) for x in aruco_gt]
    icon_size = 7
    radius = 100
    scale = g_size/3
    l_vel, r_vel = 0,0
    for i in range(3):
        apple[i] = [(x+1.5)*scale for x in apple[i]]
        lemon[i] = [(x+1.5)*scale for x in lemon[i]]
        person[i] = [(x+1.5)*scale for x in person[i]]
        g.DrawCircle(person[i], radius, fill_color='grey')
    for i in range(3):
        g.DrawCircle(apple[i], icon_size, fill_color='red')
        g.DrawCircle(lemon[i], icon_size, fill_color='yellow')
        g.DrawCircle(person[i], icon_size, fill_color='blue')
    for i in range(10):
        aruco[i] = [(x+1.5)*scale for x in aruco[i]]
        g.DrawCircle(aruco[i], icon_size, fill_color='white')
    pointer = g.DrawPoint((0,0), icon_size, color='black')
    while True:
        img = pibot.get_image()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgbytes = cv2.imencode(".png", img)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        event, values = window.read(timeout=10)
        mouse = values['-GRAPH-']
        if event == '-GRAPH-':
            if mouse == (None, None):
                continue
            print(mouse)
            g.DeleteFigure(pointer)
            pointer = g.DrawPoint(mouse, icon_size, color='black')
            if mouse[1] < 300:
                if mouse[0] < 300:
                    v1,v2 = 0,0 #stop
                elif mouse[0] > 300:
                    v1,v2 = 1,0 #straight
            elif mouse[1] > 300:
                if mouse[0] < 300:
                    v1,v2 = 0,1
                elif mouse[0] > 300:
                    v1,v2 = 0,-1
            l_vel, r_vel = pibot.set_velocity([v1,v2])
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
    window.close()
main()
