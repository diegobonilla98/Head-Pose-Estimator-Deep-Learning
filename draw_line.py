import cv2
from math import cos, sin, radians
import numpy as np


def line(frame, middle, pan_angle, tilt_angle):
    start = middle
    mag = 50

    angle = pan_angle
    rot_mat_pan = np.array([[cos(radians(angle)), sin(radians(angle))],
                            [-sin(radians(angle)), cos(radians(angle))]])
    angle = tilt_angle
    rot_mat_tilt = np.array([[cos(radians(angle)), sin(radians(angle))],
                            [-sin(radians(angle)), cos(radians(angle))]])
    pan_init_coord = np.array([[50], [0]])
    tilt_init_coord = np.array([[0], [50]])
    end_pan = tuple(np.matmul(rot_mat_pan, pan_init_coord).reshape((2, )).astype(int))
    end_tilt = tuple(np.matmul(rot_mat_tilt, tilt_init_coord).reshape((2, )).astype(int))

    frame = cv2.line(frame, start, end_pan, (255, 0, 0), 2)
    frame = cv2.line(frame, start, end_tilt, (0, 0, 255), 2)
    return frame
