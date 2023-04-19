import cv2
import os
import shutil
from PIL import Image
from pathlib import Path


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

def write_out_frames(frames: list[Image.Image], folder_path: str):
    # フォルダを作成
    folder_path = Path(folder_path)

    # 既にあったら削除
    if os.path.exists(folder_path):
        print("Folder already exists. Removing...")
        shutil.rmtree(folder_path)
        print("Removed folder", folder_path)

    
    os.makedirs(folder_path)
    print("Created folder", folder_path)

    # フォルダ内に画像を作成
    for i, frame in enumerate(frames):
        img_path = Path(folder_path, f"{i}.jpg")
        frame.save(img_path)

    print("Created images in folder", folder_path)

    return folder_path

def compress_folder(target_path: str):
    shutil.make_archive(target_path, "zip", target_path)
    return target_path + ".zip"