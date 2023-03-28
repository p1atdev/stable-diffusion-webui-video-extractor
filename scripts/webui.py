import os
import tempfile
from typing import Dict
import threading
import queue
from PIL import Image
import shutil
from pathlib import Path
import time

import gradio as gr

from modules import script_callbacks

from tool.extractor import VideoExtractor
from tool.interrogator import WD14Tagger, unload_wd14tagger
from tool.predictor import LaionAestheticPredictor, unload_laion_aesthetic_predictor
from common import TaggerModelType, LaionAestheticModelType
from utils import get_video_length, get_video_frames, write_out_frames, compress_folder

WD14TAGGER_MODELS: Dict[str, TaggerModelType] = {
    "faster": "wd14-vit-v2",
    "slower": "wd14-swinv2-v2"
}
AESTHETIC_MODELS: Dict[str, LaionAestheticModelType] = {
    "sac+logos+ava1-l14-linearMSE": "sac+logos+ava1-l14-linearMSE",
    "ava+logos-l14-linearMSE": "ava+logos-l14-linearMSE"
}

CURRENT_STATE = {
    "video_path": None,
    "extracted": [], # list[Image.Image]
    "excluded": [] # list[Image.Image]
}

LAST_PROGRESSION = 0

def on_single_video_set(video_path: str):
    print("Video set to ", video_path) 
    if video_path == "" or video_path is None:
        return ""
    
    length = get_video_length(video_path)
    frames = get_video_frames(video_path)

    message = f"Video length: {length:.2f} seconds, {frames} frames"
    print(message)

    return message

def on_single_preview_btn_clicked(
        video_path: str, 
        step_of_frames: int,
    ):
    
    print("Showing preview of ", video_path)

    if not os.path.exists(video_path):
        print("Video file not found")
        return ["Video file not found", None]
    
    try:
        frames = VideoExtractor.get_frames(video_path, step_of_frames, 12)

        frame_images: list[Image.Image] = []
        for frame in frames:
            if frame is None:
                break
            frame_images.append(frame)

        return [f"Showed preview of {video_path}", frame_images]
    except Exception as e:
        print(e)
        return [f"Error: {e}", None]

def on_single_extract_btn_clicked(
        video_path: str,
        step_of_frames: int,
        tagging_model_type: str,
        ban_word_text: str,
        ban_word_threshold: float,
        aesthetic_model_name: str,
        min_aesthetic: float,
        max_aesthetic: float
    ):
    print("Extracting frames from ", video_path)

    if not os.path.exists(video_path):
        print("Video file not found")
        return ["Video file not found", None]
    
    ban_word_tags: Dict[str, float] = {}
    for tag in  [tag.strip().replace(" ", "_") for tag in ban_word_text.split(",") if tag.strip() != ""]:
        ban_word_tags[tag] = ban_word_threshold 

    try:
        print("Extracting frames...")

        extracted_frames_queue = queue.Queue()
        excluded_frames_queue = queue.Queue()

        frame_queue = queue.Queue()

        num_processed_frames = 0
        total_frames = get_video_frames(video_path) // step_of_frames

        frame_getter_done = threading.Event()

        global LAST_PROGRESSION

        def process_worker():
            print("Loading WD 14 Tagger...")
            wd14tagger = WD14Tagger(WD14TAGGER_MODELS[tagging_model_type])
            
            print("Loading Laion Aesthetic Predictor...")
            predictor = LaionAestheticPredictor()

            while True:
                idx, frame = frame_queue.get()
                if frame is None:
                    break

                aesthetic_score = predictor.predict(aesthetic_model_name, frame)

                print(f"Frame {idx} aesthetic score: {aesthetic_score}")

                if wd14tagger.any_match(frame, ban_word_tags) or aesthetic_score < min_aesthetic or aesthetic_score > max_aesthetic:
                    excluded_frames_queue.put((idx, frame))
                else:
                    extracted_frames_queue.put((idx, frame))

        def frame_getter():
            nonlocal num_processed_frames

            frames = VideoExtractor.get_frames(video_path, step_of_frames)

            for frame, frame_index in frames:
                if frame is None:
                    break
                frame_queue.put((frame_index, frame), block=True)
                num_processed_frames += 1

            frame_queue.put((None, None))
            frame_getter_done.set()

        processing_thread = threading.Thread(target=process_worker)
        processing_thread.start()

        frame_getter_thread = threading.Thread(target=frame_getter)
        frame_getter_thread.start()

        extracted_frames = {}
        excluded_frames = {}

        while not frame_getter_done.is_set() or len(extracted_frames) + len(excluded_frames) < total_frames:
            # print(f"Extracted frames: {len(extracted_frames)}")
            current_progression = (len(extracted_frames) + len(excluded_frames)) / total_frames * 100
            if current_progression - LAST_PROGRESSION > 0.01:
                print("{:.2f} % proceeded".format(current_progression))
            LAST_PROGRESSION = current_progression

            # これを消すとマジでなぜか動かない。マジで。
            print("Extracted frames: ", len(extracted_frames))
            print("Excluded frames: ", len(excluded_frames))

            try:
                frame_index, extracted_frame = extracted_frames_queue.get(
                    block=not frame_getter_done.is_set(), timeout=1
                )
                extracted_frames[frame_index] = extracted_frame
            except queue.Empty:
                pass

            try:
                frame_index, excluded_frame = excluded_frames_queue.get(
                    block=not frame_getter_done.is_set(), timeout=1
                )
                excluded_frames[frame_index] = excluded_frame
            except queue.Empty:
                pass

        frame_getter_thread.join()
        processing_thread.join()

        extracted_frames = [extracted_frames[i] for i in sorted(extracted_frames)]
        excluded_frames = [excluded_frames[i] for i in sorted(excluded_frames)]

        print("Extracting frames completed!")

        print("Extracted frames: ", len(extracted_frames))
        print("Excluded frames: ", len(excluded_frames))

        global CURRENT_STATE
        CURRENT_STATE["video_path"] = video_path
        CURRENT_STATE["extracted"] = extracted_frames
        CURRENT_STATE["excluded"] = excluded_frames
        LAST_PROGRESSION = 0

        if len(extracted_frames) >= 30 or len(excluded_frames) >= 30:
            print("Too many frames to show. Showing only 30")
            return [
                f"Too many frames to show. Showing only 50. Extracting frames from {video_path} completed!",
                extracted_frames[:min(30, len(extracted_frames))],
                excluded_frames[:min(30, len(excluded_frames))]
            ]
        else:
            return [
                f"Extracting frames from {video_path} completed!", 
                extracted_frames, 
                excluded_frames
            ]

    except Exception as e:
        print(e)
        return [f"Error: {e}", None, None]

def on_single_download_extracted_btn_clicked():
    if "extracted" not in CURRENT_STATE:
        return ["No extracted frames", None]
    try:
        # 書き出し
        video_name = Path(CURRENT_STATE["video_path"]).stem

        # 一時ディレクトリを作成
        tmp_dir = Path(tempfile.TemporaryDirectory().name) / video_name

        # フレームを書き出す
        write_out_frames(CURRENT_STATE["extracted"], tmp_dir)

        # 圧縮して
        zip_path = compress_folder(tmp_dir.resolve())

        # パスを返す
        return ["Compressing finished! Download from below area.", zip_path, "## Download from here ↓"]
    except Exception as e:
        print(e)
        return [f"Error: {e}", None, ""]

def on_single_download_excluded_btn_clicked():
    if "excluded" not in CURRENT_STATE:
        return ["No excluded frames", None]
    try:
        # 書き出し
        video_name = Path(CURRENT_STATE["video_path"]).stem

        # 一時ディレクトリを作成
        tmp_dir = Path(tempfile.TemporaryDirectory().name) / video_name

        # フレームを書き出す
        write_out_frames(CURRENT_STATE["excluded"], tmp_dir)

        # 圧縮して
        zip_path = compress_folder(tmp_dir.resolve())

        # パスを返す
        return ["Compressing finished! Download from below area.", zip_path, "## Download from here ↓"]
    except Exception as e:
        print(e)
        return [f"Error: {e}", None, ""]

def on_single_download_all_btn_clicked():
    if "extracted" not in CURRENT_STATE or "excluded" not in CURRENT_STATE:
        return ["No extracted frames", None]
    try:
        # 書き出し
        video_name = Path(CURRENT_STATE["video_path"]).stem

        # 一時ディレクトリを作成
        tmp_dir = Path(tempfile.TemporaryDirectory().name) / video_name

        # フレームを書き出す
        write_out_frames(CURRENT_STATE["extracted"], tmp_dir / "extracted")
        write_out_frames(CURRENT_STATE["excluded"], tmp_dir / "excluded")

        # 圧縮して
        zip_path = compress_folder(tmp_dir.resolve())

        # パスを返す
        return ["Compressing finished! Download from below area.", zip_path, "## Download from here ↓"]
    except Exception as e:
        print(e)
        return [f"Error: {e}", None, ""]

def on_common_model_unload_btn_clicked():
    unload_wd14tagger()
    unload_laion_aesthetic_predictor()

    msg = "Model unloaded"
    return [msg, msg]

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Column():
            # with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem(label="Single process"):
                        with gr.Row():
                            with gr.Column():
                                single_video_input = gr.Video(format="mp4", source="upload", label="Input", interactive=True)

                                single_video_information_md = gr.Markdown("")

                                single_preview_btn = gr.Button("Preview", variant="secondary")
                                single_extracting_btn = gr.Button("Extract", variant="primary")

                            with gr.Column():
                                single_status_text = gr.Textbox("Idle", label="Status")
                                single_extracted_gallery = gr.Gallery(label="Extracted frames").style(grid=[4], height="auto")
                                single_download_extracted_btn = gr.Button("Download extracted frames (zip)", variant="primary")

                                with gr.Accordion("Show excluded", open=False):
                                    single_excluded_gallery = gr.Gallery(label="Excluded frames").style(grid=[4], height="auto")
                                    single_donwload_excluded_btn = gr.Button("Download excluded frames (zip)", variant="secondary")
                                    single_download_all_btn = gr.Button("Download all frames (zip)", variant="secondary")

                    with gr.TabItem(label="Batch process"):
                        with gr.Row():
                            with gr.Column():
                                batch_input_dir_input = gr.Textbox(label="Input directory", max_lines=1)
                                batch_output_dir_input = gr.Textbox(label="Output directory", max_lines=1)

                                batch_preview_btn = gr.Button("Preview", variant="secondary")
                                batch_extracting_btn = gr.Button("Extract", variant="primary")

                            with gr.Column():
                                batch_process_status_text = gr.Textbox("Idle", label="Status")
                                batch_preview_gallery = gr.Gallery(label="Preview of videos to process").style(grid=[4], height="auto")

                with gr.Row():
                    with gr.Column(): 
                        with gr.Column(): 
                            common_tagging_model_type = gr.Dropdown(
                                label="Tagging model",
                                choices=list(WD14TAGGER_MODELS.keys()),
                                value="faster",
                                interactive=True
                            )

                            common_ban_word_list_input = gr.Textbox(
                                label="BAN word list (supporting only danbooru tags. separete with comma)", 
                                value="blurry, close-up", 
                                placeholder="blurry, close-up, simple background ...",
                                interactive=True
                            )
                            common_ban_word_threshold_slider = gr.Slider(
                                label="BAN word threshold (0.8 = if hit with 80% confidence, exclude it)",
                                minimum=0, 
                                maximum=1, 
                                step=0.05, 
                                value=0.5, 
                                interactive=True
                            )

                        with gr.Column(): 
                        # parameters and buttons
                            common_step_of_frames_slider = gr.Slider(
                                label="Step of frames",
                                minimum=1, 
                                maximum=600, 
                                step=1, 
                                value=60, 
                                interactive=True
                            )

                            common_aesthetic_model_name = gr.Dropdown(
                                label="Aesthetic model",
                                choices=list(AESTHETIC_MODELS.keys()),
                                value="sac+logos+ava1-l14-linearMSE",
                                interactive=True
                            )

                        with gr.Row(): 
                            common_min_aesthetic_score_slider = gr.Slider(
                                label="Minimum aesthetic score", 
                                minimum=0, 
                                maximum=10, 
                                step=0.5, 
                                value=5, 
                                interactive=True
                            )
                            common_max_aesthetic_score_slider = gr.Slider(
                                label="Maximum aesthetic score",
                                minimum=0,
                                maximum=10,
                                step=0.5,
                                value=10,
                                interactive=True
                            )

                        common_model_unload_btn = gr.Button("Unload models", variant="secondary")

                    with gr.Column():
                        common_download_area_message_md = gr.Markdown("")
                        common_file_download_area = gr.File(label="Download area", format="zip", interactive=False)
    
        single_video_input.change(
            fn=on_single_video_set, 
            inputs=[single_video_input], 
            outputs=[single_video_information_md]
        )

        single_preview_btn.click(
            fn=on_single_preview_btn_clicked,
            inputs=[
                single_video_input,
                common_step_of_frames_slider,
            ],
            outputs=[
                single_status_text,
                single_extracted_gallery,
            ]
        )

        single_extracting_btn.click(
            fn=on_single_extract_btn_clicked,
            inputs=[
                single_video_input,
                common_step_of_frames_slider,
                common_tagging_model_type,
                common_ban_word_list_input,
                common_ban_word_threshold_slider,
                common_aesthetic_model_name,
                common_min_aesthetic_score_slider,
                common_max_aesthetic_score_slider,
            ],
            outputs=[
                single_status_text,
                single_extracted_gallery,
                single_excluded_gallery
            ]
        )

        single_download_extracted_btn.click(
            fn=on_single_download_extracted_btn_clicked,
            inputs=[],
            outputs=[single_status_text, common_file_download_area, common_download_area_message_md]
        )
        single_download_extracted_btn.click(
            fn=on_single_download_extracted_btn_clicked,
            inputs=[],
            outputs=[single_status_text, common_file_download_area, common_download_area_message_md]
        )
        single_download_all_btn.click(
            fn=on_single_download_all_btn_clicked,
            inputs=[],
            outputs=[single_status_text, common_file_download_area, common_download_area_message_md]
        )

        common_model_unload_btn.click(
            fn=on_common_model_unload_btn_clicked,
            inputs=[],
            outputs=[single_status_text, batch_process_status_text]
        )
    
    return [(ui, "Video Extractor", "video_extractor")]

script_callbacks.on_ui_tabs(on_ui_tabs)