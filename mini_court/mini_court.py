import cv2
import sys
import numpy as np
from utils.coordinate_utils import (m2px, px2m)


from matplotlib.pylab import dtype
sys.path.append('../')
import constants

class MiniCourt:
    def __init__(self, frames, start_x, start_y, scale, padding, ball_detections, player_detections, homography_matrix):
        self.frames = frames
        self.start_x = start_x
        self.start_y = start_y
        self.padding = padding
        self.scale = scale
        self.ball_detections = ball_detections
        self.player_detections = player_detections
        self.homography_matrix = homography_matrix
        self.lines = []   # list of lists s.t. each sub-list is a list of 2 tuples s.t. each line defined by its start and end points which are the two tuples
        self.W_px, self.H_px = self.make_mapper() # W_px, H_px = Pixel Dimensions of Mini Court Outer dimensions

        '''
        NEED:
        - Method to convert meters to mini court coordinates
        - Method to define all lines (line defined by points of two ends)
        - Method which can take two points and draw line
        - Method to add overlay (mini court) to video frames
        '''

    def make_mapper(self):
        padding = self.padding
        scale = self.scale
        W_px, H_px = (constants.DOUBLES_BASELINE * scale) + padding * 2 ,(constants.SIDELINE * scale) + padding * 2
        return W_px, H_px
        
    def set_court_lines(self):
        '''
        - Defines all court lines and stores into self.lines.
        - For every line to be drawn, we will take the two end points of the line and convert coordinates from meters to pixels using m2px
        - Save each line's end point coordinates in meters to self.lines
        '''
        def add(start, end):
            start_x, start_y = start
            end_x, end_y = end
            self.lines.append([m2px(start_x, start_y, self.scale, self.padding), m2px(end_x, end_y, self.scale, self.padding)])

        # BASELINE - NEAR
        add([0, 0], [constants.DOUBLES_BASELINE, 0])
        # BASELINE - FAR
        add([0, constants.SIDELINE], [constants.DOUBLES_BASELINE, constants.SIDELINE])
        # DOUBLES SIDELINE - LEFT
        add([0, 0], [0, constants.  SIDELINE])
        # DOUBLES SIDELINE - RIGHT
        add([constants.DOUBLES_BASELINE, 0], [constants.DOUBLES_BASELINE, constants.SIDELINE])
        # SINGLES SIDELINE - LEFT
        add([constants.DOUBLES_ALLEY, 0], [constants.DOUBLES_ALLEY, constants.SIDELINE])
        # SINGLES SIDELINE - RIGHT
        add([constants.DOUBLES_BASELINE - constants.DOUBLES_ALLEY, 0], [constants.DOUBLES_BASELINE - constants.DOUBLES_ALLEY, constants.SIDELINE])
        # NET
        add([0, constants.SIDELINE / 2], [constants.DOUBLES_BASELINE, constants.SIDELINE / 2])
        # SERVICE LINE - NEAR
        add([constants.DOUBLES_ALLEY, constants.NO_MANS_LAND], [constants.DOUBLES_BASELINE - constants.DOUBLES_ALLEY, constants.NO_MANS_LAND])
        # SERVICE LINE - FAR
        add([constants.DOUBLES_ALLEY, constants.HALF_COURT + constants.SERVICE_BOX_LENGTH], [constants.DOUBLES_BASELINE - constants.DOUBLES_ALLEY, constants.HALF_COURT + constants.SERVICE_BOX_LENGTH])
        # CENTRE SERVICE LINE
        add([constants.DOUBLES_BASELINE/2, constants.NO_MANS_LAND], [constants.DOUBLES_BASELINE/2, constants.HALF_COURT + constants.SERVICE_BOX_LENGTH])

    def draw_lines(self, frame):
        for line in self.lines:
            start, end = line
            cv2.line(frame, (self.start_x + start[0], self.start_y + start[1]) , (self.start_x + end[0], self.start_y + end[1]), (255, 255, 0), 2)

    def add_overlay(self, frame):
        # Ensure all values are integers and within frame bounds
        overlay_start_x = int(self.start_x) - 50
        overlay_start_y = int(self.start_y) - 50
        overlay_end_x = int(self.W_px + self.start_x) + 50
        overlay_end_y = int(self.H_px + self.start_y) + 50

        # # overlay = frame.copy()
        # cv2.rectangle(frame, (overlay_start_x, overlay_start_y), (overlay_end_x, overlay_end_y), (20, 30, 30), -1)
        # alpha = 0.9
        # cv2.addWeighted(frame, 1-alpha, frame, alpha, 0, frame)
         # --- Step 1: Semi-transparent filled background ---
        overlay = frame.copy()
        cv2.rectangle(overlay,
                    (overlay_start_x, overlay_start_y),
                    (overlay_end_x, overlay_end_y),
                    (20, 30, 30), -1)   # filled dark background
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- Step 2: White border ---
        cv2.rectangle(frame,
                    (overlay_start_x, overlay_start_y),
                    (overlay_end_x, overlay_end_y),
                    (255, 255, 255), 2)   # 2 px white border
    
    def draw_ball_position(self, frame, ball_dict):
        for track_id, bbox in ball_dict.items():
            x1_reg, y1_reg, x2_reg, y2_reg = bbox
            x1_reg_mid = int((x1_reg + x2_reg) / 2)
            y1_reg_mid = int((y1_reg + y2_reg) / 2)
            point_mini_m = px2m((x1_reg_mid, y1_reg_mid), self.homography_matrix)
            x1_m, y1_m = point_mini_m[0][0]
            if x1_m > constants.DOUBLES_BASELINE or y1_m > constants.SIDELINE or x1_m < 0 or y1_m < 0:
                continue
            x1_mini_px, y1_mini_px = m2px(x1_m, y1_m, self.scale, self.padding)
            cv2.circle(frame, (x1_mini_px + self.start_x, y1_mini_px + self.start_y), 5, (0, 255, 255), -1)

    def draw_player_positions(self, frame, player_dict):
        for track_id, bbox in player_dict.items():
            x1_reg, y1_reg, x2_reg, y2_reg = bbox
            x_reg = int((x1_reg + x2_reg) / 2)
            y_reg = int(max(y1_reg, y2_reg)) 
            point_mini_m = px2m((x_reg, y_reg), self.homography_matrix)
            x1_m, y1_m = point_mini_m[0][0]
            if x1_m > constants.DOUBLES_BASELINE + 3 or y1_m > constants.SIDELINE + 3 or x1_m < -3 or y1_m < -3:
                continue
            x1_mini_px, y1_mini_px = m2px(x1_m, y1_m, self.scale, self.padding)

            if track_id == 1:
                frame = cv2.rectangle(frame, (x1_mini_px + self.start_x, y1_mini_px + self.start_y), (x1_mini_px + self.start_x + 10, y1_mini_px + self.start_y + 10), (255, 0, 255), -1)
            else:
                frame = cv2.rectangle(frame, (x1_mini_px + self.start_x, y1_mini_px + self.start_y), (x1_mini_px + self.start_x + 10, y1_mini_px + self.start_y + 10), (0, 0, 255), -1)


    def draw_mini_court(self):

        self.set_court_lines()

        # Draw lines and overlay on each frame
        output_video_frames = []
        for frame, ball_dict, player_dict in zip(self.frames, self.ball_detections, self.player_detections):
            self.add_overlay(frame)
            self.draw_lines(frame)  
            self.draw_ball_position(frame, ball_dict)
            self.draw_player_positions(frame, player_dict)
        
            output_video_frames.append(frame)
        return output_video_frames