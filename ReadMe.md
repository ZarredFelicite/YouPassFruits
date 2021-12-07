Read me for group 207

download yolov3_tiny_v5.weights from google drive folder:
https://drive.google.com/drive/folders/1wZ_5AXpbG47K2RSYtf1XL16KqXmW04jz
and put weights into Group207/network/scripts/PyTorch-YOLOv3/weights/ folder

open the gazebo and run the map

rename map to map1.txt and ensure that the map in the catkin folder and the group207 folder
(the same folder as operate.py are the same map)

open terminal to group207 folder to run either semi autonomous or autonomous waypoint selection as below.

#for semi autonomous waypoint selection:
run operate.py using python3 operate.py
rotate robot to a point where it can see three aruco markers and press enter to begin slam
click on added gui map to select waypoints to move to
manually select waypoints to get robot to move to each object and move into place


#for autonomous moving
run operate.py using python3 operate.py
rotate robot to a point where it can see three aruco markers and press enter to begin slam
press 'g' to start robot moving
press 'h' if you wish to stop robot moving and return to semi autonomous selection
