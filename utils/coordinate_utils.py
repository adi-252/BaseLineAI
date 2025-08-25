import cv2
import numpy as np

def m2px(x_m, y_m, scale, padding):
    x_px = int(x_m * scale + padding)
    y_px = int(y_m * scale + padding)
    return (x_px, y_px)

def px2m(point_px, homography_matrix):
    p = np.array([[point_px]], dtype=np.float32)  
    return cv2.perspectiveTransform(p, homography_matrix)