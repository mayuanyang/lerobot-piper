from pathlib import Path
from typing import List, Optional
import json


class CameraData:
    """A class to handle video data associated with an episode."""
    
    def __init__(self, video_path: Path, camera: str, first_frames_to_chop: int = 0):
        self.video_path = video_path
        self.camera = camera
        self.first_frames_to_chop = first_frames_to_chop

class EpisodeData:
    """A class to handle inference with trained LeRobot policies."""
    
    def __init__(self, joint_data_json_path: Path, episode_index: int, fps: int, global_index_offset: int, cameras: List[CameraData], folder: str = "", task_description: str = "Pick up the cube and place it into the container.", last_frames_to_chop: int = 0):
        self.joint_data_json_path = joint_data_json_path
        self.episode_index = episode_index
        self.fps = fps
        self.global_index_offset = global_index_offset
        self.cameras = cameras
        self.num_of_frames = 0
        self.dataset_from_index = 0
        self.folder = folder
        self.task_description = task_description
        self.last_frames_to_chop = last_frames_to_chop
