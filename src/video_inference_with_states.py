#!/usr/bin/env python3
"""
Video inference script that accepts state information for each frame.
Processes video frames with provided state data and generates heatmap visualizations.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import csv
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

# Import our custom modules
from models.transformer_diffusion.transformer_diffusion_policy import TransformerDiffusionPolicy
from models.transformer_diffusion.processor_transformer_diffusion import make_pre_post_processors
from models.transformer_diffusion.spatial_softmax import save_heatmap_visualization


class VideoHeatmapInferenceWithStates:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the video inference processor."""
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        self.policy = TransformerDiffusionPolicy.from_pretrained(model_path)
        self.policy.eval()
        self.policy.to(self.device)
        
        # Get model config
        self.config = self.policy.config
        self.state_dim = self.config.state_dim
        
        # Create preprocessor
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            dataset_stats=None,  # Will use default stats or compute on-the-fly
            add_grid_overlay=True,
            grid_overlay_cameras=["front", "right"]  # Adjust based on your camera setup
        )
        
        # Move preprocessors to device
        if isinstance(self.preprocessor, torch.nn.Module):
            self.preprocessor.to(self.device)
        if isinstance(self.postprocessor, torch.nn.Module):
            self.postprocessor.to(self.device)
            
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        print(f"Model loaded successfully!")
        print(f"Expected state dimension: {self.state_dim}")
    
    def load_state_data(self, state_file):
        """
        Load state data from a file (CSV, JSON, or NPZ).
        
        Args:
            state_file: path to state data file
            
        Returns:
            list of state arrays, one per frame
        """
        state_file = Path(state_file)
        
        if state_file.suffix.lower() == '.csv':
            states = []
            with open(state_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Convert string values to floats
                    state_values = [float(x.strip()) for x in row]
                    states.append(np.array(state_values, dtype=np.float32))
            return states
            
        elif state_file.suffix.lower() == '.json':
            with open(state_file, 'r') as f:
                data = json.load(f)
            # Assume data is a list of state arrays
            states = [np.array(state, dtype=np.float32) for state in data]
            return states
            
        elif state_file.suffix.lower() == '.npz':
            data = np.load(state_file)
            # Assume there's a key called 'states' or use the first array
            if 'states' in data:
                states = data['states']
            else:
                # Get the first array
                states = list(data.values())[0]
            # Convert to list of arrays if it's a 2D array
            if states.ndim == 2:
                states = [states[i] for i in range(states.shape[0])]
            return [np.array(state, dtype=np.float32) for state in states]
            
        else:
            raise ValueError(f"Unsupported state file format: {state_file.suffix}")
    
    def process_single_frame(self, frame, state=None, camera_name="front"):
        """
        Process a single frame with optional state information.
        
        Args:
            frame: numpy array (H, W, C) in BGR format
            state: numpy array or tensor of state information (state_dim,)
            camera_name: name of the camera for processing
            
        Returns:
            dict with processed results and heatmap
        """
        with torch.no_grad():
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transformations
            tensor_image = self.transform(pil_image)
            
            # Add batch dimension
            tensor_image = tensor_image.unsqueeze(0).to(self.device)
            
            # Handle state information
            if state is not None:
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float().to(self.device)
                else:
                    state_tensor = state.float().to(self.device)
                
                # Ensure proper shape (1, 1, state_dim)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
                elif state_tensor.dim() == 2:
                    state_tensor = state_tensor.unsqueeze(0)
                    
                # Validate state dimension
                if state_tensor.shape[-1] != self.state_dim:
                    raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {state_tensor.shape[-1]}")
            else:
                # Use zero state if none provided
                state_tensor = torch.zeros(1, 1, self.state_dim).to(self.device)
            
            # Create batch dictionary
            batch = {
                f"observation.images.{camera_name}": tensor_image,
                "observation.state": state_tensor
            }
            
            # Preprocess
            batch = self.preprocessor(batch)
            
            # Get model predictions (this will generate heatmaps internally)
            actions, spatial_outputs = self.policy.model(batch)
            
            # Postprocess actions
            actions = self.postprocessor(actions)
            
            # Generate heatmap for visualization
            # Get the vision encoder for this camera
            encoder_key = camera_name.replace('.', '_')
            if encoder_key in self.policy.model.image_encoders:
                encoder = self.policy.model.image_encoders[encoder_key]
                
                # Generate debug heatmaps
                heatmap_info = encoder.generate_debug_heatmaps(
                    batch[f"observation.images.{camera_name}"],
                    camera_name=camera_name,
                    batch_index=0,
                    timestep=0
                )
                
                return {
                    "actions": actions,
                    "spatial_outputs": spatial_outputs,
                    "heatmap_info": heatmap_info,
                    "processed_frame": tensor_image
                }
            
            return {
                "actions": actions,
                "spatial_outputs": spatial_outputs,
                "heatmap_info": None,
                "processed_frame": tensor_image
            }
    
    def process_video_with_states(self, video_path, state_file, output_dir, max_frames=None):
        """
        Process video with provided state information for each frame.
        
        Args:
            video_path: path to input video
            state_file: path to state data file (CSV, JSON, or NPZ)
            output_dir: directory to save results
            max_frames: maximum number of frames to process (None for all)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load state data
        print(f"Loading state data from: {state_file}")
        states = self.load_state_data(state_file)
        print(f"Loaded {len(states)} state entries")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Validate that we have enough states
        if len(states) < total_frames:
            print(f"Warning: Only {len(states)} states provided for {total_frames} frames")
            print("Will use zero states for remaining frames")
        
        # Output video writer for heatmap overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = output_path / "heatmap_overlay.mp4"
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=max_frames or min(total_frames, len(states)), desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames and processed_count >= max_frames:
                    break
                
                # Get state for this frame
                if frame_count < len(states):
                    state = states[frame_count]
                else:
                    state = None  # Will use zero state
                    print(f"Warning: No state data for frame {frame_count}, using zero state")
                
                # Process frame
                try:
                    result = self.process_single_frame(frame, state, camera_name="front")
                    
                    # Save heatmap visualization
                    if result["heatmap_info"] is not None:
                        heatmap_save_path = output_path / f"frame_{frame_count:06d}_heatmap.png"
                        
                        # Get original frame for overlay
                        frame_tensor = result["processed_frame"][0]  # Remove batch dimension
                        frame_numpy = frame_tensor.permute(1, 2, 0).cpu().numpy()
                        
                        # Denormalize if needed (assuming ImageNet normalization)
                        frame_numpy = np.clip(frame_numpy * 255, 0, 255).astype(np.uint8)
                        
                        # Get heatmap
                        heatmap = result["heatmap_info"]["heatmaps"][0, 0]  # First sample, first channel
                        
                        # Save heatmap visualization
                        save_heatmap_visualization(
                            heatmap, 
                            frame_numpy, 
                            heatmap_save_path,
                            point_coords=None  # Could add keypoint coordinates if needed
                        )
                        
                        # Also create overlay on original frame
                        overlay_frame = frame.copy()
                        heatmap_numpy = heatmap.cpu().numpy()
                        
                        # Resize heatmap to match frame size
                        heatmap_resized = cv2.resize(heatmap_numpy, (width, height))
                        
                        # Normalize heatmap
                        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                        
                        # Apply colormap
                        heatmap_colormap = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # Blend with original frame
                        blended = cv2.addWeighted(overlay_frame, 0.6, heatmap_colormap, 0.4, 0)
                        
                        # Write to output video
                        out_video.write(blended)
                        
                        # Save individual overlay frame
                        overlay_save_path = output_path / f"frame_{frame_count:06d}_overlay.jpg"
                        cv2.imwrite(str(overlay_save_path), blended)
                    
                    processed_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                frame_count += 1
        
        # Cleanup
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {processed_count} frames")
        print(f"Heatmap visualizations saved to: {output_path}")
        print(f"Overlay video saved to: {output_video_path}")
    
    def create_sample_state_file(self, output_file, num_frames=100, state_dim=7):
        """
        Create a sample state file for testing.
        
        Args:
            output_file: path to output file
            num_frames: number of frames/states to generate
            state_dim: dimension of state vector
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for i in range(num_frames):
                    # Generate random state values
                    state = np.random.uniform(-1, 1, state_dim)
                    writer.writerow([f"{x:.6f}" for x in state])
                    
        elif output_path.suffix.lower() == '.json':
            states = []
            for i in range(num_frames):
                state = np.random.uniform(-1, 1, state_dim).tolist()
                states.append(state)
            with open(output_path, 'w') as f:
                json.dump(states, f, indent=2)
                
        elif output_path.suffix.lower() == '.npz':
            states = np.random.uniform(-1, 1, (num_frames, state_dim)).astype(np.float32)
            np.savez(output_path, states=states)
            
        print(f"Sample state file created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video inference with heatmap visualization and state data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--state_file", type=str, help="Path to state data file (CSV, JSON, or NPZ)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--create_sample", action="store_true", help="Create a sample state file for testing")
    parser.add_argument("--sample_file", type=str, default="sample_states.csv", help="Path for sample state file")
    parser.add_argument("--sample_frames", type=int, default=100, help="Number of frames for sample file")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoHeatmapInferenceWithStates(args.model_path)
    
    if args.create_sample:
        # Create sample state file
        processor.create_sample_state_file(args.sample_file, args.sample_frames, processor.state_dim)
        return
    
    if not args.state_file:
        print("Error: --state_file is required unless using --create_sample")
        return
    
    # Process video with states
    processor.process_video_with_states(
        args.video_path, 
        args.state_file, 
        args.output_dir, 
        args.max_frames
    )


if __name__ == "__main__":
    main()
