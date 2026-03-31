import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
import numpy as np


class DiffusionSinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ObjectDetector:
    """Zero-shot object detector using YOLOWorld.

    YOLOWorld combines YOLO speed with CLIP-based open-vocabulary detection:
    you specify the target class names as text and it detects them without
    any fine-tuning — typically <30 ms/image on a single GPU.

    Configure target classes via config.detection_classes (list[str]).
    Defaults to ["object", "container"] for pick-and-place tasks.
    """

    # Shared model instance (loaded once, reused across all ObjectDetector instances)
    _shared_model = None
    _shared_classes = None

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Classes to detect — set these to your specific objects
        self.detection_classes: list[str] = list(
            getattr(config, "detection_classes", ["object", "container"])
        )

        # Confidence threshold (lower = more detections, higher = fewer false positives)
        self.conf_threshold: float = float(getattr(config, "detection_conf", 0.1))

        self._load_model()

    def _load_model(self):
        """Load YOLOWorld model (shared across instances to save GPU memory)."""
        if ObjectDetector._shared_model is None or ObjectDetector._shared_classes != self.detection_classes:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "ultralytics is required for YOLOWorld: pip install ultralytics"
                )

            print(f"Loading YOLOWorld model for zero-shot detection of: {self.detection_classes}")
            model = YOLO("yolov8x-worldv2.pt")
            model.set_classes(self.detection_classes)
            ObjectDetector._shared_model = model
            ObjectDetector._shared_classes = list(self.detection_classes)
            print("YOLOWorld model loaded.")

        self.model = ObjectDetector._shared_model

    def detect_objects_and_get_bounding_boxes(
        self, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """Detect objects using YOLOWorld and return normalised 2D bounding boxes.

        Args:
            image_tensor: (B, C, H, W) float tensor, values in [0, 1] or [0, 255].

        Returns:
            bounding_boxes: (N, 4) tensor of [x1, y1, x2, y2] boxes normalised to [0, 1].
                            N = total detections across all images in the batch.
                            Returns empty tensor when nothing is detected.
            object_types:   list[str] of length N with the detected class name for each box.
        """
        B, C, H, W = image_tensor.shape

        # Convert tensor batch to list of numpy uint8 images for ultralytics
        imgs_np = []
        img_cpu = image_tensor.detach().cpu()
        if img_cpu.max() <= 1.0 + 1e-3:
            img_cpu = (img_cpu * 255).clamp(0, 255)
        img_cpu = img_cpu.byte()  # (B, C, H, W) uint8

        for i in range(B):
            # ultralytics expects HWC numpy
            imgs_np.append(img_cpu[i].permute(1, 2, 0).numpy())

        try:
            results = self.model(imgs_np, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            print(f"YOLOWorld inference error: {e}")
            return torch.zeros((0, 4), device=image_tensor.device), []

        all_boxes: list[list[float]] = []
        all_labels: list[str] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu()  # absolute pixel coords [x1, y1, x2, y2]
            cls_ids = boxes.cls.cpu().long()
            for j in range(len(xyxy)):
                all_boxes.append(xyxy[j].tolist())
                cls_id = int(cls_ids[j].item())
                label = (
                    self.detection_classes[cls_id]
                    if cls_id < len(self.detection_classes)
                    else "unknown"
                )
                all_labels.append(label)

        if all_boxes:
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32, device=image_tensor.device)
        else:
            boxes_tensor = torch.zeros((0, 4), device=image_tensor.device)

        return boxes_tensor, all_labels
