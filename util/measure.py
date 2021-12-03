import numpy as np

class Marker:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (0.01*np.eye(2))):
        if tag in [11,12,13,14,15,16]:
            #self.covariance = (0.01*np.eye(2))
            self.covariance = (0.1*np.eye(2))
        elif tag in [17,18,19]:
            #self.covariance = (0.01*np.eye(2))
            self.covariance = (0.1*np.eye(2))
        else:
            self.covariance = covariance
        self.position = position
        self.tag = tag

class Drive:
    # Measurement of the robot wheel velocities
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov