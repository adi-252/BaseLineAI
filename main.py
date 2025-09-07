from mini_court import (MiniCourt)

from utils import(read_video, 
                  save_video, 
                  compute_homography)

from trackers import (PlayerTracker,
                      BallTracker)

from court_line_detector import CourtLineDetector

from analysis import (Analyzer)

import matplotlib.pyplot as plt

import cv2
import numpy as np



def main():
    # Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)     
    
    # Set video FPS
    fps = 24
    # Set scale
    scale = 16
    padding = 10

    # Detect players and ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl"
                                                 )
    # Interpolate ball positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Draw frame number (1-indexed) on top right corner of each frame
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    court_model_path = "models/keypoints_model_4.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Identiy and filter players based on court keypoints
    player_ids = player_tracker.identify_players(court_keypoints, player_detections)
    filtered_player_detections = player_tracker.filter_players(player_ids, player_detections)

    # Computer homography matrix
    homography_matrix = compute_homography(court_keypoints, video_frames[0])

    # Draw players and ball bboxes along with court keypoints on video frames
    output_frames = player_tracker.draw_bboxes(video_frames, filtered_player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    # Draw Mini Court
    mini_court = MiniCourt(output_frames, start_x=1600, start_y=100, scale=scale, padding=padding, player_detections=filtered_player_detections, ball_detections=ball_detections, homography_matrix=homography_matrix)
    output_frames, overlay_start_x, overlay_start_y = mini_court.draw_mini_court()
    
    analyzer = Analyzer(homography_matrix, fps)
    analyzer.create_speed_series(ball_detections, player_detections)
    analyzer.player_distance_and_radial_velocity()
    analyzer.find_impacts()
    analyzer.compute_shot_speeds()
    output_frames = analyzer.draw_analysis_box(output_frames, overlay_start_x, overlay_start_y)

    # Identify shot speeds, shot frames
    # shot_frames, shot_speeds = analyzer.identify_shots(ball_speeds, ball_coords_m)


    # Draw shot speeds, num shots, % forehand shots, % backhand shots, player_speeds

    # Save Video
    save_video(output_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
