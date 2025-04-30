import cv2

def rotate_frame(frame, angle_degree):
    height, width = frame.shape[:2]
    M = cv2.getRotationMatrix2D((width/2, height/2), angle_degree, 1.0)
    return cv2.warpAffine(frame, M, (width, height))