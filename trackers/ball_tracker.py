from ultralytics import YOLO
import cv2
import pickle 
import pandas as pd

class BallTracker:

    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        ball_detections = [x.get(1, []) for x in ball_detections]
        
        # convert into DataFrame for interpolation
        df_ball_positions = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions= df_ball_positions.bfill()

        # convert back to list of dicts
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect balls in multiple frames.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: List of dictionaries containing detected ball information for each frame.
        """
        ball_detections = []

        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict) 

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        """
        Detect balls in a single frame.

        Args:
            frame (numpy.ndarray): The video frame to process.

        Returns:
            dict: Map of detected ball id to bounding box.
        """
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        """
        Draw bounding boxes on video frames.

        Args:
            video_frames (list): List of video frames.
            ball_detections (list): List of ball detections for each frame.
            
        Returns:
            list: List of video frames with bounding boxes drawn.
        """
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw bounding boxes on the frame
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 0, 255), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames