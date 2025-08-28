from mini_court import (MiniCourt)

from utils import(read_video, 
                  save_video, 
                  compute_homography)

from trackers import (PlayerTracker,
                      BallTracker)

from court_line_detector import CourtLineDetector

from analysis.speed_analysis import (create_speed_series,
                                      identify_shots)

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
    output_frames = mini_court.draw_mini_court()

    ball_speeds, ball_coords_m, p1_speeds, p2_speeds, p1_coords_m, p2_coords_m = create_speed_series(ball_detections, player_detections, fps=fps, H=homography_matrix)

    # Identify shot speeds, shot frames
    shot_speeds, shot_frames = identify_shots(ball_speeds, ball_coords_m)

    # Draw shot speeds, num shots, % forehand shots, % backhand shots, player_speeds

    # Save Video
    save_video(output_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()


#    # Initialize speed tracking variables
#     prev_ball_m = prev_p1_m = prev_p2_m = None
#     ball_speeds = []
#     p1_speeds = []
#     p2_speeds = []

#     # Calculate speeds for each frame
#     for frame_idx, (frame, ball_dict, player_dict) in enumerate(zip(video_frames, ball_detections, filtered_player_detections)):
#         # Get ball position (center of bounding box)
    #         if ball_dict and 1 in ball_dict:
    #             ball_bbox = ball_dict[1]
    #             bx1, by1, bx2, by2 = ball_bbox
    #             ball_px = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)  # center of ball bbox
    #         else:
#             ball_px = None
            
#         # Get player positions (ground points)
#         p1_px = p2_px = None
#         if len(player_dict) >= 1:
#             player1_id = list(player_dict.keys())[0]
#             p1_bbox = player_dict[player1_id]
#             p1_px = ground_point_from_bbox(*p1_bbox)
            
#         if len(player_dict) >= 2:
#             player2_id = list(player_dict.keys())[1]
#             p2_bbox = player_dict[player2_id]
#             p2_px = ground_point_from_bbox(*p2_bbox)

#         # Convert to meters and calculate speeds
#         if ball_px is not None:
#             ball_m = to_meters(ball_px, homography_matrix)
#             ball_speed = speed_kmh(prev_ball_m, ball_m, fps)
#             ball_speeds.append(ball_speed)
#             prev_ball_m = ball_m
#         else:
#             ball_speed = 0.0
#             ball_speeds.append(ball_speed)
            
#         if p1_px is not None:
#             p1_m = to_meters(p1_px, homography_matrix)
#             p1_speed = speed_kmh(prev_p1_m, p1_m, fps)
#             p1_speeds.append(p1_speed)
#             prev_p1_m = p1_m
#         else:
#             p1_speed = 0.0
#             p1_speeds.append(p1_speed)
            
#         if p2_px is not None:
#             p2_m = to_meters(p2_px, homography_matrix)
#             p2_speed = speed_kmh(prev_p2_m, p2_m, fps)
#             p2_speeds.append(p2_speed)
#             prev_p2_m = p2_m
#         else:
#             p2_speed = 0.0
#             p2_speeds.append(p2_speed)

#         # Draw speeds on frame
#         if ball_px is not None:
#             cv2.putText(frame, f"Ball: {ball_speed:.1f} km/h", 
#                        (int(ball_px[0]), int(ball_px[1])-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
#         if p1_px is not None:
#             cv2.putText(frame, f"P1: {p1_speed:.1f} km/h", 
#                        (int(p1_px[0]), int(p1_px[1])-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
#         if p2_px is not None:
#             cv2.putText(frame, f"P2: {p2_speed:.1f} km/h", 
#                        (int(p2_px[0]), int(p2_px[1])-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

#     # Print average speeds
#     print(f"Average Ball Speed: {np.mean(ball_speeds):.1f} km/h")
#     print(f"Average Player 1 Speed: {np.mean(p1_speeds):.1f} km/h")
#     print(f"Average Player 2 Speed: {np.mean(p2_speeds):.1f} km/h")
    
#     # Output all ball speeds
#     print("\n=== ALL BALL SPEEDS ===")
#     print("Frame | Ball Speed (km/h)")
#     print("------|------------------")
#     for frame_idx, speed in enumerate(ball_speeds):
#         print(f"{frame_idx+1:5d} | {speed:8.1f}")
    
#     # Save ball speeds to a file for analysis
#     with open("output_videos/ball_speeds.txt", "w") as f:
#         f.write("Frame,Ball_Speed_kmh\n")
#         for frame_idx, speed in enumerate(ball_speeds):
#             f.write(f"{frame_idx+1},{speed:.1f}\n")
    
#     print(f"\nBall speeds saved to: output_videos/ball_speeds.txt")
#     print(f"Total frames processed: {len(ball_speeds)}")
#     print(f"Max ball speed: {np.max(ball_speeds):.1f} km/h")
#     print(f"Min ball speed: {np.min(ball_speeds):.1f} km/h")

