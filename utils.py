import cv2

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        return duration
    else:
        return 0
    
def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count
    else:
        return 0
