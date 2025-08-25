
import numpy as np


def create_speed_series(coords_m, fps):
    '''
    Create a time series of speed measurements for the ball and players.
    '''
    coords_m = np.asarray(coords_m, float)
    numFrames = coords_m.shape[0]

    if len(coords_m) < 2:
        return np.zeros(len(coords_m))
    dists = np.linalg.norm(coords_m[1:] - coords_m[:-1], axis=1)
    speeds = dists * fps * 3.6
    print(speeds)
    return speeds

