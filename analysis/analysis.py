import cv2
import numpy as np
from utils.coordinate_utils import px2m
import numpy as np, cv2
from scipy.signal import find_peaks
import copy


class Analyzer:

    def __init__(self, H, fps):
        self.H = H
        self.fps = fps
        self.COURT_W = 10.97  # m (doubles width) — change to 8.23 if you modeled singles
        self.COURT_L = 23.77  # m
        self.ball_speeds = []
        self.ball_coords_m = []
        self.p1_speeds = []
        self.p2_speeds = []
        self.p1_coords_m = []
        self.p2_coords_m = []
        self.d_b_to_closest_player = []
        self.radial_v = []
        self.closest_player = []
        self.shot_placements = []
        self.cur_shot_placement = {'frame': 0, 'left': 0, 'middle': 0, 'right': 0}

    def px2m_point(self, pt_px):
        p = np.array([[pt_px]], np.float32)     # (1,1,2)
        return cv2.perspectiveTransform(p, self.H)[0,0]  # (x_m, y_m)

    def m2px_points(self, pts_m):
        Hinv = np.linalg.inv(self.H)
        pts = np.asarray(pts_m, np.float32).reshape(-1,1,2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1,2)

    def baseline_lines_px(self):
        # Endpoints of near (y=0) and far (y=L) baselines in PIXELS
        near = self.m2px_points([(0,0), (self.COURT_W,0)])     # [[x0,y0],[x1,y1]]
        far  = self.m2px_points([(0,self.COURT_L), (self.COURT_W,self.COURT_L)])
        return near, far

    def y_on_line_at_x(self, p0, p1, x):
        (x0,y0), (x1,y1) = p0, p1
        if abs(x1-x0) < 1e-6:  # vertical-ish; fallback
            return (y0+y1)*0.5
        t = (x - x0) / (x1 - x0)
        return y0 + t*(y1 - y0)

    def bbox_bottom_mid(self, b):
        x1,y1,x2,y2 = b
        return ((x1+x2)/2.0, y2)

    def clamp_ground_px(self, bbox, is_near, near_line_px, far_line_px):
        xmid, ybot = self.bbox_bottom_mid(bbox)
        y_near = self.y_on_line_at_x(near_line_px[0], near_line_px[1], xmid)
        y_far  = self.y_on_line_at_x(far_line_px[0],  far_line_px[1],  xmid)

        low, high = (min(y_near, y_far), max(y_near, y_far))
        # image Y increases downward
        if is_near:
            y = min(ybot, high)   # don't go below the lower (on-screen) baseline
        else:
            y = max(ybot, low)    # don't go above the upper (on-screen) baseline
        return (xmid, y)

    def clamp_ball_px(self, bbox, near_line_px, far_line_px):
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

        y_near = self.y_on_line_at_x(near_line_px[0], near_line_px[1], cx)
        y_far  = self.y_on_line_at_x(far_line_px[0],  far_line_px[1],  cx)

        low  = min(y_near, y_far)
        high = max(y_near, y_far)
        cy_clamped = np.clip(cy, low, high)  # robust even if lines cross/slant

        return (cx, cy_clamped)

    def create_speed_series(self, ball_positions, player_positions):
        """
        ball_positions: list over frames of {ball_id: (x1,y1,x2,y2)}
        player_positions: list over frames of {track_id: bbox}  (assume ids 1 & 2)
        """
        near_line_px, far_line_px = self.baseline_lines_px()

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
                self.ball_coords_m.append(self.ball_coords_m[-1] if self.ball_coords_m else (np.nan, np.nan))
            else:
                # pick the first/only ball bbox
                (_, bb) = next(iter(ball_dict.items()))
                cx, cy = self.clamp_ball_px(bb, near_line_px, far_line_px)
                self.ball_coords_m.append(tuple(self.px2m_point((cx,cy))))

            # --- players ---
            p_dict = player_positions[t]
            # default carry-forward if missing
            self.p1_coords_m.append(self.p1_coords_m[-1] if self.p1_coords_m else (np.nan,np.nan))
            self.p2_coords_m.append(self.p2_coords_m[-1] if self.p2_coords_m else (np.nan,np.nan))
            if p_dict:
                if near_id in p_dict:
                    p1_px = self.clamp_ground_px(p_dict[near_id], True,  near_line_px, far_line_px)
                    self.p1_coords_m[-1] = tuple(self.px2m_point(p1_px))
                if far_id in p_dict:
                    p2_px = self.clamp_ground_px(p_dict[far_id],  False, near_line_px, far_line_px)
                    self.p2_coords_m[-1] = tuple(self.px2m_point(p2_px))

        self.ball_coords_m = np.asarray(self.ball_coords_m, float)
        self.p1_coords_m   = np.asarray(self.p1_coords_m, float)
        self.p2_coords_m   = np.asarray(self.p2_coords_m, float)

        # Ball instantaneous speed series (km/h)
        if len(self.ball_coords_m) < 2:
            self.ball_speeds = np.zeros(len(self.ball_coords_m))
        else:
            dists = np.linalg.norm(self.ball_coords_m[1:] - self.ball_coords_m[:-1], axis=1)  # m/frame
            self.ball_speeds = np.r_[0.0, dists * self.fps * 3.6]              # km/h

        # Player instantaneous speed series (km/h)
        self.p1_speeds = np.r_[0.0, np.linalg.norm(self.p1_coords_m[1:] - self.p1_coords_m[:-1], axis=1) * self.fps * 3.6] if len(self.p1_coords_m) >= 2 else np.zeros(len(self.p1_coords_m))
        self.p2_speeds = np.r_[0.0, np.linalg.norm(self.p2_coords_m[1:] - self.p2_coords_m[:-1], axis=1) * self.fps * 3.6] if len(self.p2_coords_m) >= 2 else np.zeros(len(self.p2_coords_m))

    def moving_average(self, x, k=3):
        k = max(1, int(k))
        return x if k == 1 else np.convolve(x, np.ones(k)/k, mode="same")

    def player_distance_and_radial_velocity(self):
        """
        Calculate the distance and radial velocity between the ball and a player.
        """
        b = np.asarray(self.ball_coords_m, float)
        p1 = np.asarray(self.p1_coords_m, float)
        p2 = np.asarray(self.p2_coords_m, float)

        # ball distance to each player
        d1 = np.full(len(b), np.inf, float)
        d2 = np.full(len(b), np.inf, float)

        d1 = np.linalg.norm(b - p1, axis=1)
        d2 = np.linalg.norm(b - p2, axis=1)

        # distance to nearest player and whom that is
        d = np.minimum(d1, d2)   
        who = np.where(d1 <= d2, 1, 2)  

        # smoothing
        self.d_b_to_closest_player = self.moving_average(d, k=5)

        # radial velocity
        self.radial_v = np.zeros_like(self.d_b_to_closest_player)
        self.radial_v[1:] = (self.d_b_to_closest_player[1:] - self.d_b_to_closest_player[:-1]) * self.fps * 3.6   # km/h
        self.closest_player = np.where(d1 <= d2, 1, 2)

        
    def find_impacts(self):
        T = len(self.d_b_to_closest_player)

        candidates= []
        prev_is_local_min = False

        for t in range(1, T - 1):
            is_local_min = (self.d_b_to_closest_player[t] < self.d_b_to_closest_player[t - 1]) and (self.d_b_to_closest_player[t] <= self.d_b_to_closest_player[t + 1])
            sign_flip    = (self.radial_v[t - 1] < 0) and (self.radial_v[t] >= 0)
            if sign_flip and prev_is_local_min:
                candidates.append(t)
            prev_is_local_min = is_local_min
        self.candidates = candidates

    def compute_shot_speeds(self):
        POST_WINDOW_S = 1       # seconds after impact to search for peak
        ROBUST_PCT    = 95         # robust peak = percentile to reduce single-frame spikes
        MIN_SHOT_KMH = 70

        v   = np.asarray(self.ball_speeds, float)     # km/h
        fps = float(self.fps)
        T   = len(v)
        W   = max(1, int(round(POST_WINDOW_S * fps)))

        shots = []
        cand = self.candidates

        for i, t0 in enumerate(cand):
            # end by window
            t1_win = min(T - 1, t0 + W)
            # also stop before the next impact (avoid blending two shots)
            t1_next = cand[i+1] - 1 if i + 1 < len(cand) else t1_win
            t1 = min(t1_win, t1_next)
            if t1 < t0:
                continue

            seg = v[t0:t1+1]  # km/h segment
            if seg.size == 0 or not np.isfinite(seg).any():
                continue

            peak_kmh = float(np.percentile(seg, ROBUST_PCT))
            t_peak   = int(t0 + int(np.nanargmax(seg)))

            if peak_kmh < MIN_SHOT_KMH:
                continue   # skip this candidate

            # which player hit (nearest at impact frame)
            player = int(self.closest_player[t0]) if len(self.closest_player) == T else None

            # set shot_placement using the frame of the next impact
            next_impact_frame = None
            if i + 1 < len(cand):
                next_impact_frame = cand[i + 1]
            if self.shot_placements:
                curr_shot_placement = copy.deepcopy(self.shot_placements[-1])
                curr_shot_placement['frame'] = t0
            else:
                curr_shot_placement = {
                    'frame': t0,
                    'left': 0,
                    'middle': 0,
                    'right': 0
                }

            if next_impact_frame is not None and next_impact_frame < T:
                ball_x_at_next_impact = self.ball_coords_m[next_impact_frame][0]
                if 1.37 <= ball_x_at_next_impact < (1.37 + 8.23/3):
                    curr_shot_placement['left'] += 1
                elif (1.37 + 8.23/3) <= ball_x_at_next_impact < (1.37 + (8.23/3) * 2):
                    curr_shot_placement['middle'] += 1
                elif (1.37 + (8.23/3) * 2) <= ball_x_at_next_impact < (1.37 + (8.23/3) * 3):
                    curr_shot_placement['right'] += 1

            self.shot_placements.append(curr_shot_placement)


            shots.append({
                "t": t0,                    # impact frame
                "t_peak": t_peak,           # frame of max speed in window
                "player": player,           # 1 or 2 (if available)
                "peak_kmh": peak_kmh,       # per-shot reported speed
                "speed_at_impact_kmh": float(v[t0])
            })

        self.shots = shots

        return shots

    def draw_box(self, frame, overlay_start_x, overlay_start_y):
        # Draw a semi-transparent rectangle for the analysis box
        overlay = frame.copy()
        self.analysis_box_x1 = overlay_start_x
        self.analysis_box_y1 = overlay_start_y + 40
        analysis_box_x2 = self.analysis_box_x1 + 300
        analysis_box_y2 = self.analysis_box_y1 + 350

        cv2.rectangle(overlay, (self.analysis_box_x1, self.analysis_box_y1), (analysis_box_x2, analysis_box_y2), (20, 30, 30), -1)  # filled dark background
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw the white border
        cv2.rectangle(frame, (self.analysis_box_x1, self.analysis_box_y1), (analysis_box_x2, analysis_box_y2), (255, 255, 255), 2)

    def draw_shot_speed(self, frame, frame_number):
        """
        Draw shot speed information and current player speeds on the frame.
        Returns True if a shot was displayed, False otherwise.
        """
        # Always show current player speeds
        if frame_number < len(self.p1_speeds) and frame_number < len(self.p2_speeds):
            p_speed_text_x = self.analysis_box_x1 + 10
            p_speed_text_y = self.analysis_box_y1 + 30
            
            # Draw current player speeds
            p1_speed = self.p1_speeds[frame_number]
            p2_speed = self.p2_speeds[frame_number]
            
            # Player 1 speed
            p1_text = f"P1: {p1_speed:.1f} km/h"
            cv2.putText(frame, p1_text, (p_speed_text_x, p_speed_text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Player 2 speed
            p2_text = f"P2: {p2_speed:.1f} km/h"
            cv2.putText(frame, p2_text, (p_speed_text_x, p_speed_text_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Update p_speed_text_y for shot information
            p_speed_text_y += 60
            
        # Show shot information if available
        shot_displayed = False
        if hasattr(self, 'shots') and self.shots:
            # Find shots that are currently being displayed (within a time window)
            DISPLAY_WINDOW_FRAMES = 30  # Show shot info for 30 frames after impact
            
            for shot in self.shots:
                # Check if this shot should be displayed at current frame
                if shot['t'] <= frame_number <= shot['t'] + DISPLAY_WINDOW_FRAMES:
                    # Calculate position for text
                    text_x = self.analysis_box_x1 + 10
                    shot_text_y = p_speed_text_y
                    
                    # Draw shot information
                    shot_text = f"Shot P{shot['player']}: {shot['peak_kmh']:.1f} km/h"
                    cv2.putText(frame, shot_text, (text_x, shot_text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw impact frame indicator
                    impact_text = f"Frame: {shot['t']}"
                    cv2.putText(frame, impact_text, (text_x, shot_text_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    shot_displayed = True

    def draw_shot_placement_distribution(self, frame, frame_number):
        '''
        Args:
            frame: the frame to draw the shot distribution on
            frame_number: the frame number to draw the shot distribution on
        Returns:
            None
        
        Description:
            Draw the shot distribution between (left, center, right) on the frame.
        '''
        cur_shot_placement = self.cur_shot_placement

        # Check if we need to update the current shot placement
        if self.shot_placements and len(self.shot_placements) > 0:
            first_shot_placement = self.shot_placements[0]
            
            if frame_number >= first_shot_placement['frame']:
                # It's time to use this shot placement
                self.cur_shot_placement = copy.deepcopy(first_shot_placement)
                self.shot_placements.pop(0)  # Remove the first element

        L = int(self.cur_shot_placement.get("left", 0))
        M = int(self.cur_shot_placement.get("middle", 0))
        R = int(self.cur_shot_placement.get("right", 0))

        total = max(1, L + M + R)
        p = np.array([L/total, M/total, R/total], float)

        left_rect_width = int(p[0] * 260)
        middle_rect_width = int(p[1] * 260)
        right_rect_width = int(p[2] * 260)

        cv2.putText(frame, "Shot Placement Distribution)", (self.analysis_box_x1 + 10, self.analysis_box_y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)                    
        box_y = self.analysis_box_y1 + 160
        box_x = self.analysis_box_x1 + 20

        # main rectangle
        cv2.rectangle(frame, (box_x, box_y), (box_x + 260, box_y + 40), (255, 255, 255), 2)
        # left rectangle
        cv2.rectangle(frame, (box_x + 2, box_y + 2), (box_x + 2 + left_rect_width, box_y + 38), (0, 0, 255), -1)
        # middle rectangle
        cv2.rectangle(frame, (box_x + 2 + left_rect_width, box_y + 2), (box_x + 2 + left_rect_width + middle_rect_width, box_y + 38), (0, 255, 0), -1)
        # right rectangle
        cv2.rectangle(frame, (box_x + 2 + left_rect_width + middle_rect_width, box_y + 2), (box_x + 2 + left_rect_width + middle_rect_width + right_rect_width, box_y + 38), (255, 0, 0), -1)

        # Add text labels in the middle of each box
        text_y = box_y + 25  # Center vertically in the box

        # Left box label
        if left_rect_width > 10:  # Only draw if box is wide enough
            left_text_x = box_x + 2 + left_rect_width // 2
            cv2.putText(frame, "L", (left_text_x - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Middle box label
        if middle_rect_width > 10:  # Only draw if box is wide enough
            middle_text_x = box_x + 2 + left_rect_width + middle_rect_width // 2
            cv2.putText(frame, "M", (middle_text_x - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Right box label
        if right_rect_width > 10:  # Only draw if box is wide enough
            right_text_x = box_x + 2 + left_rect_width + middle_rect_width + right_rect_width // 2
            cv2.putText(frame, "R", (right_text_x - 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return None


    def draw_shot_speed_distribution(self, frame, frame_number):
        shots = self.shots
        if not shots:
            return
        return 

              
    def draw_analysis_box(self, frames, overlay_start_x, overlay_start_y):
        output_frames = []
        for i, frame in enumerate(frames):
            self.draw_box(frame, overlay_start_x, overlay_start_y)
            self.draw_shot_speed(frame, i)  
            self.draw_shot_placement_distribution(frame, i)
            output_frames.append(frame)
        return output_frames



