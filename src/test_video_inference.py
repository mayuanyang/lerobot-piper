"""
Test script for video inference functionality.
This script tests the video inference components without requiring a trained model.
"""

import numpy as np
import cv2
from pathlib import Path

# Import our video inference class
try:
    from .video_inference import VideoInference, create_sample_joint_states
except ImportError:
    # Fallback for when running as a script directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from video_inference import VideoInference, create_sample_joint_states

def test_preprocess_frame():
    """Test the frame preprocessing functionality."""
    print("Testing frame preprocessing...")
    
    # Create a sample frame (simulating a video frame)
    sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create video inference instance
    video_inference = VideoInference("src/model_output")
    
    # Test preprocessing
    processed_frame = video_inference.preprocess_frame(sample_frame)
    
    print(f"Original frame shape: {sample_frame.shape}")
    print(f"Processed frame shape: {processed_frame.shape}")
    print(f"Processed frame dtype: {processed_frame.dtype}")
    print(f"Processed frame range: [{processed_frame.min():.3f}, {processed_frame.max():.3f}]")
    
    # Verify the preprocessing
    assert processed_frame.shape == (84, 84, 3), f"Expected shape (84, 84, 3), got {processed_frame.shape}"
    assert processed_frame.dtype == np.float32, f"Expected dtype float32, got {processed_frame.dtype}"
    assert 0 <= processed_frame.min() <= processed_frame.max() <= 1, "Values should be normalized to [0, 1]"
    
    print("✓ Frame preprocessing test passed")
    return True

def test_sample_joint_states():
    """Test the sample joint states generation."""
    print("\nTesting sample joint states generation...")
    
    # Generate sample joint states
    joint_states = create_sample_joint_states(10)
    
    print(f"Generated {len(joint_states)} joint states")
    print(f"First joint state: {joint_states[0]}")
    print(f"Last joint state: {joint_states[-1]}")
    
    # Verify the joint states
    assert len(joint_states) == 10, f"Expected 10 joint states, got {len(joint_states)}"
    assert all(len(state) == 7 for state in joint_states), "Each joint state should have 7 values"
    assert all(state.dtype == np.float32 for state in joint_states), "Joint states should be float32"
    
    print("✓ Sample joint states test passed")
    return True

def test_video_inference_class():
    """Test the VideoInference class initialization."""
    print("\nTesting VideoInference class initialization...")
    
    # Create video inference instance
    video_inference = VideoInference("src/model_output")
    
    print(f"Model path: {video_inference.model_path}")
    print(f"Dataset ID: {video_inference.dataset_id}")
    print(f"Device: {video_inference.device}")
    
    print("✓ VideoInference class initialization test passed")
    return True

def main():
    """Run all tests."""
    print("Video Inference Component Tests")
    print("=" * 40)
    
    try:
        # Run all tests
        test_video_inference_class()
        test_preprocess_frame()
        test_sample_joint_states()
        
        print("\n" + "=" * 40)
        print("All tests passed! ✓")
        print("\nThe video inference components are working correctly.")
        print("You can now use video_inference.py with a trained model.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
