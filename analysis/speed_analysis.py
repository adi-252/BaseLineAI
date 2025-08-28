
import cv2
import numpy as np
from BaseLineAI.utils.coordinate_utils import px2m
from BaseLineAI import constants
import numpy as np, cv2
from scipy.signal import find_peaks


COURT_W = 10.97  # m (doubles width) — change to 8.23 if you modeled singles
COURT_L = 23.77  # m

def px2m_point(pt_px, H):
    p = np.array([[pt_px]], np.float32)     # (1,1,2)
    return cv2.perspectiveTransform(p, H)[0,0]  # (x_m, y_m)

def m2px_points(pts_m, H):
    Hinv = np.linalg.inv(H)
    pts = np.asarray(pts_m, np.float32).reshape(-1,1,2)
    return cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)

def baseline_lines_px(H):
    # Endpoints of near (y=0) and far (y=L) baselines in PIXELS
    near = m2px_points([(0,0), (COURT_W,0)], H)     # [[x0,y0],[x1,y1]]
    far  = m2px_points([(0,COURT_L), (COURT_W,COURT_L)], H)
    return near, far

def y_on_line_at_x(p0, p1, x):
    (x0,y0), (x1,y1) = p0, p1
    if abs(x1-x0) < 1e-6:  # vertical-ish; fallback
        return (y0+y1)*0.5
    t = (x - x0) / (x1 - x0)
    return y0 + t*(y1 - y0)

def bbox_bottom_mid(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, y2)

def clamp_ground_px(bbox, is_near, near_line_px, far_line_px):
    xmid, ybot = bbox_bottom_mid(bbox)
    y_near = y_on_line_at_x(near_line_px[0], near_line_px[1], xmid)
    y_far  = y_on_line_at_x(far_line_px[0],  far_line_px[1],  xmid)

    low, high = (min(y_near, y_far), max(y_near, y_far))
    # image Y increases downward
    if is_near:
        y = min(ybot, high)   # don't go below the lower (on-screen) baseline
    else:
        y = max(ybot, low)    # don't go above the upper (on-screen) baseline
    return (xmid, y)

def clamp_ball_px(bbox, near_line_px, far_line_px):
    x1,y1,x2,y2 = bbox
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

    y_near = y_on_line_at_x(near_line_px[0], near_line_px[1], cx)
    y_far  = y_on_line_at_x(far_line_px[0],  far_line_px[1],  cx)

    low  = min(y_near, y_far)
    high = max(y_near, y_far)
    cy_clamped = np.clip(cy, low, high)  # robust even if lines cross/slant

    return (cx, cy_clamped)

def create_speed_series(ball_positions, player_positions, fps, H):
    """
    ball_positions: list over frames of {ball_id: (x1,y1,x2,y2)}
    player_positions: list over frames of {track_id: bbox}  (assume ids 1 & 2)
    """
    near_line_px, far_line_px = baseline_lines_px(H)

    ball_m, p1_m, p2_m = [], [], []

    # Decide which track is near vs far once (based on first frame bottoms)
    first_players = next((d for d in player_positions if len(d)>=2), None)
    assert first_players is not None, "Need two player boxes"
    items = sorted(first_players.items())  # [(id,bbox), (id,bbox)]
    (id_a, box_a), (id_b, box_b) = items[0], items[1]
    # larger pixel y2 is nearer to the camera
    is_a_near = box_a[3] > box_b[3]
    near_id, far_id = (id_a, id_b) if is_a_near else (id_b, id_a)


    # create coords_m arrays for ball and both players
    for t in range(len(ball_positions)):
        # --- ball ---
        ball_dict = ball_positions[t]
        if not ball_dict: 
            ball_m.append(ball_m[-1] if ball_m else (np.nan, np.nan))
        else:
            # pick the first/only ball bbox
            (_, bb) = next(iter(ball_dict.items()))
            cx, cy = clamp_ball_px(bb, near_line_px, far_line_px)
            ball_m.append(tuple(px2m_point((cx,cy), H)))

        # --- players ---
        p_dict = player_positions[t]
        # default carry-forward if missing
        p1_m.append(p1_m[-1] if p1_m else (np.nan,np.nan))
        p2_m.append(p2_m[-1] if p2_m else (np.nan,np.nan))
        if p_dict:
            if near_id in p_dict:
                p1_px = clamp_ground_px(p_dict[near_id], True,  near_line_px, far_line_px)
                p1_m[-1] = tuple(px2m_point(p1_px, H))
            if far_id in p_dict:
                p2_px = clamp_ground_px(p_dict[far_id],  False, near_line_px, far_line_px)
                p2_m[-1] = tuple(px2m_point(p2_px, H))

    ball_m = np.asarray(ball_m, float)
    p1_m   = np.asarray(p1_m, float)
    p2_m   = np.asarray(p2_m, float)

    # Ball instantaneous speed series (km/h)
    if len(ball_m) < 2:
        speeds = np.zeros(len(ball_m))
    else:
        dists = np.linalg.norm(ball_m[1:] - ball_m[:-1], axis=1)  # m/frame
        speeds = np.r_[0.0, dists * fps * 3.6]                    # km/h

    # Player instantaneous speed series (km/h)
    p1_speeds = np.r_[0.0, np.linalg.norm(p1_m[1:] - p1_m[:-1], axis=1) * fps * 3.6] if len(p1_m) >= 2 else np.zeros(len(p1_m))
    p2_speeds = np.r_[0.0, np.linalg.norm(p2_m[1:] - p2_m[:-1], axis=1) * fps * 3.6] if len(p2_m) >= 2 else np.zeros(len(p2_m))

    return speeds, ball_m, p1_speeds, p2_speeds, p1_m, p2_m

def identify_shots(ball_speeds, ball_coords_m):
    shot_frames = []
    shot_speeds = []

    peaks, _ = find_peaks(ball_speeds, distance = 20)  # height in km/h, distance in frames

    for i in peaks:
        if ball_speeds[i] > 80:
            shot_frames.append(i)
            shot_speeds.append(ball_speeds[i])

    return shot_frames, shot_speeds
