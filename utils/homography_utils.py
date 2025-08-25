import cv2
from matplotlib import scale
import constants
import numpy as np


# calculate homography matrix
def compute_homography(court_keypoints, frame):
    '''
    Compute the homography matrix based on court keypoints.
    The court keypoints are expected to be in the following order:
        point 0 - Bottom left corner
        point 1 - Bottom right corner
        point 2 - Top left corner
        point 3 - Top right corner

    Args:
        court_keypoints (list): List of court keypoints.
        frame (numpy.ndarray): The frame from which the homography is computed.
        
    Returns:
        numpy.ndarray: The computed homography matrix.
    '''
    src_pts = court_keypoints[0:8].reshape(-1, 2)
    dst_pts = np.array([
    [0.0 , 0.0],
    [constants.DOUBLES_BASELINE, 0.0],
    [0.0,    constants.SIDELINE],
    [constants.DOUBLES_BASELINE, constants.SIDELINE]],
    dtype=np.float32)

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    return homography_matrix


    # # # Test Homogrqaphy - Using src_pts and homography_matrix to test if dst_pts can be reconstructed
    # # reconstructed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), homography_matrix)
    # # reconstructed_pts = reconstructed_pts.reshape(-1, 2)
    # # print("Original Points: ", src_pts)
    # # print("Reconstructed Points: ", reconstructed_pts)

    # # # Computing Warped Perspective
    # # desired_width = 800
    # # desired_height = 1700
    # # scale_x = desired_width / 10.97
    # # scale_y = desired_height / 23.76
    # # scale = min(scale_x, scale_y)  # Use the smaller scale to fit both dimensions within the image
    # # dst_pts = dst_pts * scale
    # # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # # imgOutput = cv2.warpPerspective(frame, matrix, (int(desired_width), int(desired_height)))
    # # cv2.imshow("Original Frame", frame)
    # # cv2.imshow("Warped Perspective", imgOutput)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()