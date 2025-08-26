
import numpy as np

from BaseLineAI.utils.coordinate_utils import px2m


def create_speed_series(ball_positions, fps, homography_matrix):
    '''
    Create a time series of speed measurements for the ball and players.
    '''
    coords_m = []
    for ball_dict in ball_positions:
        for track_id, bbox in ball_dict.items():
            x1, y1, x2, y2 = bbox
            ball_px = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # center of ball bbox
            ball_m = px2m(ball_px, homography_matrix)
            coords_m.append((ball_m[0][0][0], ball_m[0][0][1]))
    coords_m = np.array(coords_m)
    if len(coords_m) < 2:
        return np.zeros(len(coords_m))
    dists = np.linalg.norm(coords_m[1:] - coords_m[:-1], axis=1)
    speeds = dists * fps * 3.6
    return speeds, coords_m

# def identify_shots(speeds, coords_m,  ):


# def identify_shots(speeds, coords_m):
