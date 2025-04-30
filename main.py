import os
import cv2
import yaml
from src.motion_analyzer import MotionAnalyzer
from src.ai_pipeline import MultiVideoPredictor

def main(
    video_dir,
    cfg_path="./cfg/default.yaml"
    ):
    
    ls_video_path = [os.path.join(video_dir, i) for i in os.listdir(video_dir) if i.endswith(".mp4")]
    
    cfg = yaml.safe_load(open(cfg_path, "r"))
    
    motionAnalyzer = MotionAnalyzer(cfg)
    
    
    isDraw = cfg["BASIC"]["DRAW"]
    if isDraw:
        cv2.namedWindow("Feature Motion", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Feature Motion", 800, 600)
        cv2.namedWindow("Prev Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prev Frame", 800, 600)
    
    video_process_thread = motionAnalyzer.video_capture(ls_video_path, draw=isDraw)
    fps = motionAnalyzer.fps
    
    # run ai model
    if cfg["MODEL"]["ENABLE"]:
        predictor = MultiVideoPredictor(
            cfg,
            num_threads=cfg["MODEL"]["NUM_THREADS"]
            )
        display_queue = predictor.buffer
        ai_process_thread = predictor.muti_predict(motionAnalyzer.buffer, draw=isDraw)
    else:
        display_queue = motionAnalyzer.buffer
    
    while True:
        
        video_dict = display_queue.get()
        
        if video_dict is None:
            break
        
        prev_frame = video_dict["prev_frame"]
        draw_frame = video_dict["draw_frame"]
        
        if draw_frame is not None:
            cv2.imshow("Feature Motion", draw_frame)
            cv2.imshow("Prev Frame", prev_frame)
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == 27:
            break
        
    video_process_thread.join(timeout=5)
    if cfg["MODEL"]["ENABLE"]:
        ai_process_thread.join(timeout=5)
    
    cv2.destroyAllWindows()     

if __name__ == "__main__":
    
    video_dir = "./data"
    cfg_path = "./cfg/default.yaml"
    
    main(video_dir, cfg_path=cfg_path)