from ultralytics import YOLO
import cv2
import pickle 

class PlayerTracker:

    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect players in multiple frames.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: List of dictionaries containing detected player information for each frame.
        """
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict) 

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        """
        Detect players in a single frame.

        Args:
            frame (numpy.ndarray): The video frame to process.

        Returns:
            dict: Map of detected player id to bounding box.
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue  # Skip boxes without a track ID
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draw bounding boxes on video frames.

        Args:
            video_frames (list): List of video frames.
            player_detections (list): List of player detections for each frame.
            
        Returns:
            list: List of video frames with bounding boxes drawn.
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes on the frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames

    def identify_players(self, court_keypoints, player_detections):
        """
        - Filter players based on average proximity to court keypoints.
        - Using first frame. Calculate average euclidean distance of player bbox center to court keypoints.
        - Keep two closest players to every keypoint on average.

        Args:
            court_keypoints (list): List of court keypoints.
            player_detections (list): List of player detections for each frame.
            
        Returns:
            list: List of player IDs of the closest players to court keypoints.
        """

        first_frame_players = player_detections[0]
        filtered_player_detections = []
        keypoint_pairs = [(court_keypoints[i], court_keypoints[i+1]) 
                      for i in range(0, len(court_keypoints), 2)]

        for track_id, bbox in first_frame_players.items():
            x1, y1, x2, y2 = bbox
            # print("Player ID:", track_id, "BBox:", bbox)
            player_center = ((x1 + x2) / 2, (y1 + y2) / 2)

            total_distance = 0
            # calculate average distance to court keypoints
            for i, kp in enumerate(keypoint_pairs):
                distance = ((player_center[0] - kp[0]) ** 2 + (player_center[1] - kp[1]) ** 2) ** 0.5
                total_distance += distance

            # average distance to all keypoints
            avg_distance = total_distance / 13  # Assuming 14 keypoints
            filtered_player_detections.append((track_id, bbox, avg_distance))
        
        filtered_player_detections = filtered_player_detections[:2]
        filtered_player_ids = [track_id for track_id, _, _ in filtered_player_detections]
        return filtered_player_ids

    def filter_players(serlf, player_ids, player_detections):
        """
        Filter players based on provided player IDs.
        Args:
            player_ids (list): List of player IDs to filter.
            player_detections (list): List of player detections for each frame.
            
        Returns:
            list: List of filtered player detections.
        """
        player_detections_copy = player_detections.copy()
        for i, frame in enumerate(player_detections):
            temp_dict = {}
            for player_id, bbox in frame.items():
                if player_id in player_ids:
                    temp_dict[player_id] = bbox
            player_detections[i] = temp_dict

        return player_detections