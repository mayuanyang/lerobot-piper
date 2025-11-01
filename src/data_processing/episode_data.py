from pathlib import Path
from typing import List, Optional


class CameraData:
    """A class to handle video data associated with an episode."""
    
    def __init__(self, video_path: Path, camera: str):
        self.video_path = video_path
        self.camera = camera

class EpisodeData:
    """A class to handle inference with trained LeRobot policies."""
    
    def __init__(self, joint_data_json_path: Path, episode_index: int, fps: int, global_index_offset: int, cameras: List[CameraData]):
        self.joint_data_json_path = joint_data_json_path
        self.episode_index = episode_index
        self.fps = fps
        self.global_index_offset = global_index_offset
        self.cameras = cameras
        
        
