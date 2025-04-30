import cv2
import numpy as np
import queue
import threading
from .utils import rotate_frame
import os

class MotionAnalyzer():
    def __init__(self, cfg):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.cfg = cfg
        
        self.buffer = queue.Queue(maxsize=1024)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None

    def extract_keypoints(self, frame):
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    def grid_select_keypoints(self, keypoints, descriptors, image_shape, grid_size=(4, 4), max_keypoints_per_grid=30):
        h, w = image_shape[:2]
        grid_h, grid_w = grid_size

        selected_keypoints = []
        selected_descriptors = []

        cell_h = h // grid_h
        cell_w = w // grid_w

        if keypoints is None or descriptors is None:
            return None, None
        
        # 每個格子分開挑
        for gy in range(grid_h):
            for gx in range(grid_w):
                x_start = gx * cell_w
                y_start = gy * cell_h
                x_end = x_start + cell_w
                y_end = y_start + cell_h

                # 抓出這格內的keypoints
                cell_keypoints = []
                cell_descriptors = []
                for kp, des in zip(keypoints, descriptors):
                    if x_start <= kp.pt[0] < x_end and y_start <= kp.pt[1] < y_end:
                        cell_keypoints.append(kp)
                        cell_descriptors.append(des)

                # 按 response 排序，取最多 max_keypoints_per_grid 個
                sorted_cell = sorted(zip(cell_keypoints, cell_descriptors), key=lambda x: x[0].response, reverse=True)
                selected = sorted_cell[:max_keypoints_per_grid]

                for kp, des in selected:
                    selected_keypoints.append(kp)
                    selected_descriptors.append(des)

        return selected_keypoints, np.array(selected_descriptors)

    def match_frames(self, kp1_info, kp2_info):
        kp1, des1 = kp1_info
        kp2, des2 = kp2_info
        
        if des1 is None or des2 is None:
            return []
        
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches

    def get_homography(self, matches, prev_kp, cur_kp):
        dst_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([cur_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if len(src_pts) < 4:
            return [], None
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        matches_mask = mask.ravel().tolist()
        filtered_matches = [m for i, m in enumerate(matches) if matches_mask[i]]
        matches = filtered_matches
        return filtered_matches, H
    
    def determine_status(self, matches, kp1, kp2):
        # 根據matches數量或分佈分析是否靜止或重複
        if len(matches) == 0:
            return "No matches"
        
        distances = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            distance = np.sqrt(dx*dx + dy*dy)
            distances.append(distance)

        distances = np.array(distances)
        mean_pixel_distance = np.mean(distances)

        # 設一個合理的像素移動閾值
        movement_threshold = 2

        if mean_pixel_distance > movement_threshold:
            return f"moving({mean_pixel_distance:.2f})"  # 有在移動
        else:
            return "static"  # 幾乎靜止  
    
    def run(self, ls_video_path:list):
        prev_frame = None
        prev_kp = None
        prev_des = None
        
        if self.cfg["MANUAL_CORRECT"]["ENABLE"]:
            # manual create homography matrix
            pts_src = np.float32(self.cfg["MANUAL_CORRECT"]["PTS_SRC"])
            pts_dst = np.float32(self.cfg["MANUAL_CORRECT"]["PTS_DST"])
            
            shift_x = self.cfg["MANUAL_CORRECT"]["SHIFT_X"]
            shift_y = self.cfg["MANUAL_CORRECT"]["SHIFT_Y"]
            pts_dst = pts_dst + np.array([shift_x, shift_y], dtype=np.float32)
            
            H, _ = cv2.findHomography(pts_src, pts_dst)
        
        
        img_count = 0
        for video_path in ls_video_path:
            cap = cv2.VideoCapture(video_path)
            video_name = os.path.basename(video_path)
            frame_count_in_video = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                height, width = frame.shape[:2]
                if self.cfg["MANUAL_CORRECT"]["ENABLE"]:
                    frame = cv2.warpPerspective(frame, H, (width, height))
                
                if prev_frame is None:
                    prev_kp, prev_des = self.extract_keypoints(frame)
                    prev_kp, prev_des = self.grid_select_keypoints(prev_kp, prev_des, (height, width))
                    prev_frame = frame
                    continue
                
                else:
                    kp, des = self.extract_keypoints(frame)
                    kp, des = self.grid_select_keypoints(kp, des, (height, width))
                    matches = self.match_frames(
                        kp1_info=(prev_kp, prev_des),
                        kp2_info=(kp, des)
                    )
                    
                    matches, H2prevFrame = self.get_homography(matches, prev_kp, kp)
                    
                    # draw direction
                    draw_frame = frame.copy()
                    if self.draw:
                        if self.cfg["MANUAL_CORRECT"]["ENABLE"]:
                            vectors = []
                            
                        for m in matches:
                            pt1 = tuple(map(int, prev_kp[m.queryIdx].pt))
                            pt2 = tuple(map(int, kp[m.trainIdx].pt))
                            
                            if self.cfg["MANUAL_CORRECT"]["ENABLE"]:
                                vec = np.array(pt2) - np.array(pt1)
                                vectors.append(vec)
                            
                            cv2.circle(draw_frame, pt1, 3, (0,255,0), -1)
                            cv2.arrowedLine(draw_frame, pt1, pt2, (0, 255, 0), 2, tipLength=0.1)
                        
                        if self.cfg["MANUAL_CORRECT"]["ENABLE"]:    
                            draw_frame = rotate_frame(draw_frame, self.cfg["MANUAL_CORRECT"]["ANGLE_DEGREE"])
                            angles = [np.arctan2(v[1], v[0]) for v in vectors]
                            angle_deg = np.degrees(angles)
                            angle_diff = np.std(angle_deg)  # 標準差代表「方向分散程度」
                            
                            lengths = [np.linalg.norm(v) for v in vectors]

                            print(f"角度標準差：{angle_diff:.2f}°, 長度標準差:{int(np.std(lengths))}")
                        
                        str_status = self.determine_status(matches, prev_kp, kp)
                        cv2.putText(draw_frame, str_status, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                    else:
                        draw_frame = None
                    
                    video_dict = {
                        "frame_id": img_count,
                        "frame": frame.copy(),
                        "kp": kp,
                        "matches": matches,
                        "H2prevFrame": H2prevFrame,
                        "prev_kp": prev_kp,
                        "prev_frame": prev_frame,
                        "status": str_status,
                        "draw_frame": draw_frame.copy(),
                        "video_name": video_name,
                        "frame_count_in_video": frame_count_in_video
                    }
                    
                    self.buffer.put(video_dict)
                    prev_kp = kp
                    prev_frame = draw_frame if self.draw else frame
                    prev_des = des
                
                img_count += 1
                frame_count_in_video += 1
                    
        
        # end
        self.buffer.put(None)
        
    def video_capture(self, ls_video_path:list, draw=False):
        self.draw = draw
        
        # get fps
        cap = cv2.VideoCapture(ls_video_path[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.fps = fps
        
        video_process_thread = threading.Thread(target=self.run, args=(ls_video_path, ))
        video_process_thread.start()
        return video_process_thread