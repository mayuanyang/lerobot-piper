#!/usr/bin/env python3
"""
Video inference script with heatmap visualization for transformer diffusion model.
Processes video frames and applies spatial softmax heatmaps for analysis.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

# Import our custom modules
from models.transformer_diffusion.transformer_diffusion_policy import TransformerDiffusionPolicy
from models.transformer_diffusion.processor_transformer_diffusion import make_pre_post_processors
from models.transformer_diffusion.spatial_softmax import save_heatmap_visualization


class VideoHeatmapInference:
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
        print(f"Model config: {self.config}")
    
    def process_single_frame(self, frame, state=None, camera_name="front"):
        """
        Process a single frame and generate heatmap visualization.
        
        Args:
            frame: numpy array (H, W, C) in BGR format
            state: numpy array or tensor of state information (1, state_dim)
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
            else:
                # Use zero state if none provided
                state_tensor = torch.zeros(1, 1, self.config.state_dim).to(self.device)
            
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
    
    def process_video(self, video_path, output_dir, max_frames=None):
        """
        Process video and generate heatmap visualizations for each frame.
        
        Args:
            video_path: path to input video
            output_dir: directory to save results
            max_frames: maximum number of frames to process (None for all)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
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
        
        # Output video writer for heatmap overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = output_path / "heatmap_overlay.mp4"
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        with tqdm(total=max_frames or total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if max_frames and processed_count >= max_frames:
                    break
                
                # Process frame
                try:
                    result = self.process_single_frame(frame, camera_name="front")
                    
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
                    continue
                
                frame_count += 1
        
        # Cleanup
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {processed_count} frames")
        print(f"Heatmap visualizations saved to: {output_path}")
        print(f"Overlay video saved to: {output_video_path}")
    
    def benchmark_performance(self, video_path, num_frames=100):
        """
        Benchmark the inference performance on video frames.
        
        Args:
            video_path: path to input video
            num_frames: number of frames to process for benchmarking
        """
        import time
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        times = []
        frame_count = 0
        
        print(f"Benchmarking performance on {num_frames} frames...")
        
        with torch.no_grad():
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                try:
                    # Process frame
                    result = self.process_single_frame(frame, camera_name="front")
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        avg_time = np.mean(times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0
                        print(f"Processed {frame_count}/{num_frames} frames - Avg time: {avg_time:.4f}s, FPS: {fps:.2f}")
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
        
        cap.release()
        
        # Calculate statistics
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print("\n" + "="*50)
            print("PERFORMANCE BENCHMARK RESULTS")
            print("="*50)
            print(f"Total frames processed: {len(times)}")
            print(f"Average processing time: {avg_time:.4f} ± {std_time:.4f} seconds")
            print(f"Average FPS: {fps:.2f}")
            print(f"Min time: {np.min(times):.4f}s")
            print(f"Max time: {np.max(times):.4f}s")
            
            return {
                "avg_time": avg_time,
                "std_time": std_time,
                "fps": fps,
                "min_time": np.min(times),
                "max_time": np.max(times),
                "total_frames": len(times)
            }
        else:
            print("No frames were successfully processed!")
            return None


def main():
    parser = argparse.ArgumentParser(description="Video inference with heatmap visualization")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark_frames", type=int, default=100, help="Number of frames for benchmarking")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoHeatmapInference(args.model_path)
    
    if args.benchmark:
        # Run benchmark
        processor.benchmark_performance(args.video_path, args.benchmark_frames)
    else:
        # Process video
        processor.process_video(args.video_path, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()
