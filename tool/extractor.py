from typing import Iterator, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

def frame_to_pil_image(frame: np.ndarray) -> Image.Image:
    # BGRからRGBに変換します。
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # numpy.ndarrayからPILのImageに変換します。
    pil_image = Image.fromarray(frame_rgb)
    return pil_image

class VideoExtractor():
    def get_frames(video_path: str, frame_interval: int = 1, max_frames: Optional[int] = None,) -> Iterator[Tuple[Image.Image, int]]:        # 動画を読み込む
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open the video file {video_path}")
            return

        frame_count = 0
        captured_frame_count = 0

        # フレーム数を取得
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                yield frame_to_pil_image(frame), captured_frame_count
                captured_frame_count += 1

                if max_frames is not None and captured_frame_count >= max_frames:
                    break

            frame_count += 1
            

        # 動画を解放する
        cap.release()