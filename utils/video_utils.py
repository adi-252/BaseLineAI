
import cv2

def read_video(video_path):
    """
    Reads a video file and returns the frames.

    Args:
        video_path (str): Path to the video file.
    
    Returns:
        list: List of frames from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Saves a list of video frames to a video file.

    Args:
        output_video_frames (list): List of frames to be saved.
        output_video_path (str): Path to the output video file.
    """
    if not output_video_frames:
        print("No frames to save.")
        return

    # Get the width, height, and FPS from the first frame
    height, width, _ = output_video_frames[0].shape
    fps = 24  # You can change this to the desired FPS

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()