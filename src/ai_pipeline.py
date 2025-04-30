from ultralytics import YOLO
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
import numpy as np
import time

class MultiVideoPredictor:
    def __init__(self, cfg, num_threads=1):
        self.cfg = cfg
        self.models = [YOLO(self.cfg["MODEL"]["PATH"]) for _ in range(num_threads)]
        self.buffer = queue.Queue(maxsize=1024)
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.names = self.models[0].names
        self.img_count = 0
        
    def _predict(self, video_dict, img_count):
        """單張圖片預測"""
        img = video_dict["frame"]
        model = self.models[img_count % len(self.models)]  # 分配模型
        results = model(img, verbose=False)
        video_dict["result"] = results[0]
        return video_dict
    
    def _callback(self, fut):
        """把預測結果加到 buffer"""
        try:
            video_dict = fut.result()
            result = video_dict["result"]
            
            if not self.draw:
                while video_dict["frame_id"] != self.img_count:
                    time.sleep(0.1)
                self.buffer.put(video_dict)
                self.img_count += 1
                return
            
            H2prevFrame = video_dict["H2prevFrame"]
            frame = video_dict["draw_frame"]
            if frame is None:
                frame = video_dict["frame"]
            prev_frame = video_dict["prev_frame"]
            
            boxes = result.boxes.xyxy.cpu().numpy()  # 邊界框（bounding box）
            scores = result.boxes.conf.cpu().numpy()  # 置信度
            labels = result.boxes.cls.cpu().numpy()   # 分類結果（label）
            
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                polygons = result.masks.xy
            else:
                masks = [None for _ in range(len(boxes))]
                polygons = [None for _ in range(len(boxes))]
            
            # 繪製邊界框  
            for box, score, label, mask, polygon in zip(boxes, scores, labels, masks, polygons):
                color = self.cfg["DISTRESS_COLOR"][self.names[int(label)]]
                x1, y1, x2, y2 = box
                
                if mask is not None:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8)
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[:, :] = color
                    alpha = 0.2  # 半透明程度，0.0～1.0之間
                    frame = np.where(mask[:, :, None], (frame * (1 - alpha) + color_mask * alpha).astype(np.uint8), frame)
                
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
                cv2.putText(frame, f"{self.names[int(label)]}", (int(x1), int(y1) + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
                cv2.putText(frame, f" {score:.2f}", (int(x1), int(y1) + 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

                # transform polygons x,y to last frame position
                if polygon is not None:
                    color = (0, 0, 255) # 紅色代表過去的
                    poly_np = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)  # 轉成(N,1,2)
                    transformed_poly = cv2.perspectiveTransform(poly_np, H2prevFrame)  # 用H轉換
                    transformed_poly = transformed_poly.reshape(-1, 2)  # 還原成(N,2)
                    cv2.polylines(prev_frame, [transformed_poly.astype(np.int32)], True, color, 5)
        
            video_dict["draw_frame"] = frame
            video_dict["prev_frame"] = prev_frame
            
            while video_dict["frame_id"] != self.img_count:
                time.sleep(0.1)
            self.buffer.put(video_dict)
            self.img_count += 1
            
        except Exception as e:
            print(f"Predict failed: {e}")
            video_dict["result"] = None
            self.buffer.put(video_dict)

    def run(self, src_queue):
        img_count = 0
        while True:
            video_dict = src_queue.get()
            
            if video_dict is None:
                self.buffer.put(None)
                break
            
            future = self.executor.submit(self._predict, video_dict, img_count)
            future.add_done_callback(self._callback)

            img_count += 1
        
        print(f"Finished {img_count} images")
    
    def muti_predict(self, src_queue, draw=False):
        self.draw = draw
        predict_thread = threading.Thread(target=self.run, args=(src_queue, ))
        predict_thread.start()
        return predict_thread
            