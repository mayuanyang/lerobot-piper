import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any
import time


class YOLOSegmentation:
    """A class to handle YOLO segmentation using webcam frames."""
    
    def __init__(self, model_path: str = "yolov8n-seg.pt", confidence_threshold: float = 0.5):
        """
        Initialize the YOLO segmentation engine.
        
        Args:
            model_path (str): Path to the YOLO segmentation model or model name
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """
        Load the YOLO segmentation model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading YOLO segmentation model: {self.model_path}")
            print(f"Using device: {self.device}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            print("YOLO segmentation model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def segment_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Perform segmentation on a single frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            dict: Segmentation results including masks, boxes, classes, etc.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Extract results
            result = results[0]  # Get first (and typically only) result
            
            # Prepare output dictionary
            output = {
                "boxes": result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([]),
                "scores": result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([]),
                "classes": result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([]),
                "masks": result.masks.data.cpu().numpy() if result.masks is not None else None,
                "names": result.names if hasattr(result, 'names') else {},
                "frame_shape": frame.shape
            }
            
            return output
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None
    
    def visualize_segmentation(self, frame: np.ndarray, segmentation_result: Dict[str, Any]) -> np.ndarray:
        """
        Visualize segmentation results on the frame.
        
        Args:
            frame (np.ndarray): Original frame
            segmentation_result (dict): Results from segment_frame
            
        Returns:
            np.ndarray: Frame with segmentation visualization
        """
        if segmentation_result is None:
            return frame
            
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Get segmentation data
        boxes = segmentation_result.get("boxes", np.array([]))
        scores = segmentation_result.get("scores", np.array([]))
        classes = segmentation_result.get("classes", np.array([]))
        masks = segmentation_result.get("masks", None)
        names = segmentation_result.get("names", {})
        
        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            score = scores[i]
            cls = int(classes[i])
            label = names.get(cls, f"Class {cls}")
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label}: {score:.2f}"
            cv2.putText(vis_frame, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw masks if available
        if masks is not None:
            # Create a transparent overlay for masks
            mask_overlay = np.zeros_like(vis_frame, dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                # Convert mask to binary
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # Resize mask to frame size if needed
                if binary_mask.shape != vis_frame.shape[:2]:
                    binary_mask = cv2.resize(binary_mask, (vis_frame.shape[1], vis_frame.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask
                color = self._get_color(i)
                colored_mask = np.zeros_like(vis_frame, dtype=np.uint8)
                colored_mask[binary_mask == 1] = color
                
                # Add to overlay
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.3, 0)
            
            # Blend overlay with original frame
            vis_frame = cv2.addWeighted(vis_frame, 1.0, mask_overlay, 0.5, 0)
        
        return vis_frame
    
    def _get_color(self, index: int) -> Tuple[int, int, int]:
        """
        Generate a color based on index for mask visualization.
        
        Args:
            index (int): Index to generate color for
            
        Returns:
            tuple: RGB color values
        """
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        return colors[index % len(colors)]
    
    def get_class_masks(self, segmentation_result: Dict[str, Any], target_class: int) -> Optional[np.ndarray]:
        """
        Extract masks for a specific class.
        
        Args:
            segmentation_result (dict): Results from segment_frame
            target_class (int): Class ID to extract masks for
            
        Returns:
            np.ndarray: Binary mask for the target class, or None if not found
        """
        if segmentation_result is None:
            return None
            
        classes = segmentation_result.get("classes", np.array([]))
        masks = segmentation_result.get("masks", None)
        
        if masks is None or len(classes) == 0:
            return None
            
        # Find indices of the target class
        target_indices = np.where(classes == target_class)[0]
        
        if len(target_indices) == 0:
            return None
            
        # Combine masks for the target class
        combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for idx in target_indices:
            combined_mask = np.logical_or(combined_mask, masks[idx] > 0.5)
            
        return combined_mask.astype(np.uint8)


def main():
    """Main function demonstrating YOLO segmentation with webcam."""
    print("YOLO Segmentation Demo")
    print("=" * 30)
    
    # Initialize YOLO segmentation
    yolo_seg = YOLOSegmentation("yolov8n-seg.pt")  # Using nano model for speed
    
    # Load model
    if not yolo_seg.load_model():
        print("Failed to load YOLO model")
        return
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Starting webcam segmentation. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Perform segmentation
            start_time = time.time()
            result = yolo_seg.segment_frame(frame)
            inference_time = time.time() - start_time
            
            if result is not None:
                # Visualize results
                vis_frame = yolo_seg.visualize_segmentation(frame, result)
                
                # Add FPS counter
                fps = 1.0 / inference_time if inference_time > 0 else 0
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('YOLO Segmentation', vis_frame)
            else:
                # Show original frame if segmentation failed
                cv2.imshow('YOLO Segmentation', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Segmentation interrupted by user")
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
