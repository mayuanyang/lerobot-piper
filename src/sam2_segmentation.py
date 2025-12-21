import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
import time
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


class SAM2Segmentation:
    """A class to handle SAM2 segmentation using webcam frames."""
    
    def __init__(self, model_path: str = "src/sam2.1_b.pt", model_type: str = "vit_b", confidence_threshold: float = 0.5):
        """
        Initialize the SAM2 segmentation engine.
        
        Args:
            model_path (str): Path to the SAM2 segmentation model
            model_type (str): Type of SAM model (vit_h, vit_l, vit_b)
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.predictor = None
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
        Load the SAM2 segmentation model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading SAM2 segmentation model: {self.model_path}")
            print(f"Using device: {self.device}")
            
            # Register and load the SAM model
            self.model = sam_model_registry[self.model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)
            self.model.eval()
            
            # Create predictor
            self.predictor = SamPredictor(self.model)
            
            print("SAM2 segmentation model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def segment_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Perform segmentation on a single frame using SAM2.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            dict: Segmentation results including masks, boxes, classes, etc.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for predictor
            self.predictor.set_image(frame_rgb)
            
            # For demonstration, we'll generate some sample points for segmentation
            # In a real implementation, you might get these from user input or another detection model
            h, w = frame_rgb.shape[:2]
            
            # Generate sample points in a grid pattern
            input_points = []
            input_labels = []
            
            # Create a simple grid of points
            grid_size = 5
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int(w * (i + 1) / (grid_size + 1))
                    y = int(h * (j + 1) / (grid_size + 1))
                    input_points.append([x, y])
                    # Label as foreground (1) for demonstration
                    input_labels.append(1)
            
            if len(input_points) > 0:
                input_points = np.array(input_points)
                input_labels = np.array(input_labels)
                
                # Run predictor
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
                
                # Prepare output dictionary
                output = {
                    "boxes": np.array([]),  # SAM doesn't directly provide boxes
                    "scores": scores if scores is not None else np.array([]),
                    "classes": np.array([0] * len(scores)) if scores is not None else np.array([]),  # All as class 0
                    "masks": masks,
                    "names": {0: "object"},
                    "frame_shape": frame.shape,
                    "input_points": input_points,
                    "input_labels": input_labels
                }
                
                return output
            else:
                # Return empty result if no points
                return {
                    "boxes": np.array([]),
                    "scores": np.array([]),
                    "classes": np.array([]),
                    "masks": None,
                    "names": {},
                    "frame_shape": frame.shape
                }
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
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
        masks = segmentation_result.get("masks", None)
        scores = segmentation_result.get("scores", np.array([]))
        input_points = segmentation_result.get("input_points", None)
        input_labels = segmentation_result.get("input_labels", None)
        
        # Draw masks if available
        if masks is not None:
            # Create a transparent overlay for masks
            mask_overlay = np.zeros_like(vis_frame, dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                # Convert mask to binary
                if mask.dtype != np.uint8:
                    binary_mask = (mask > 0.5).astype(np.uint8)
                else:
                    binary_mask = mask
                
                # Resize mask to frame size if needed
                if binary_mask.shape != vis_frame.shape[:2]:
                    binary_mask = cv2.resize(binary_mask, (vis_frame.shape[1], vis_frame.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask
                color = self._get_color(i)
                colored_mask = np.zeros_like(vis_frame, dtype=np.uint8)
                if len(binary_mask.shape) == 2:
                    colored_mask[binary_mask == 1] = color
                else:
                    colored_mask[binary_mask[:, :, 0] == 1] = color
                
                # Add to overlay
                alpha = 0.3 + (0.2 * i)  # Different transparency for each mask
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, alpha, 0)
            
            # Blend overlay with original frame
            vis_frame = cv2.addWeighted(vis_frame, 1.0, mask_overlay, 0.5, 0)
        
        # Draw input points if available
        if input_points is not None and input_labels is not None:
            for point, label in zip(input_points, input_labels):
                x, y = int(point[0]), int(point[1])
                color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for foreground, red for background
                cv2.circle(vis_frame, (x, y), 5, color, -1)
                cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 2)  # White border for visibility
        
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


def main():
    """Main function demonstrating SAM2 segmentation with webcam."""
    print("SAM2 Segmentation Demo")
    print("=" * 30)
    
    # Initialize SAM2 segmentation
    sam_seg = SAM2Segmentation("src/sam2.1_b.pt", "vit_b")
    
    # Load model
    if not sam_seg.load_model():
        print("Failed to load SAM2 model")
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
            result = sam_seg.segment_frame(frame)
            inference_time = time.time() - start_time
            
            if result is not None:
                # Visualize results
                vis_frame = sam_seg.visualize_segmentation(frame, result)
                
                # Add FPS counter
                fps = 1.0 / inference_time if inference_time > 0 else 0
                cv2.putText(vis_frame, f"SAM2 FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('SAM2 Segmentation', vis_frame)
            else:
                # Show original frame if segmentation failed
                cv2.imshow('SAM2 Segmentation', frame)
            
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
