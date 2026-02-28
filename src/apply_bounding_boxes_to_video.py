#!/usr/bin/env python3
"""
Script to apply precomputed bounding boxes to a video and save the annotated video.

Supports both 3D and 2D object detection results:
- 3D mode: Draws 3D bounding boxes as wireframe cubes
- 2D mode: Draws 2D bounding boxes as rectangles

The script automatically detects the type of bounding boxes in the JSON file and applies
the appropriate visualization.

Usage:
    For 3D detection results:
    python apply_bounding_boxes_to_video.py \\
        --json_path /path/to/3d_bounding_boxes.json \\
        --video_path /path/to/video.mp4 \\
        --output_path /path/to/output_video.mp4

    For 2D detection results:
    python apply_bounding_boxes_to_video.py \\
        --json_path /path/to/2d_bounding_boxes.json \\
        --video_path /path/to/video.mp4 \\
        --output_path /path/to/output_video.mp4
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import math
import random


def rotate_xyz(point, pitch, yaw, roll):
    """Rotate a 3D point by the given angles"""
    x0, y0, z0 = point
    x1 = x0
    y1 = y0 * math.cos(pitch) - z0 * math.sin(pitch)
    z1 = y0 * math.sin(pitch) + z0 * math.cos(pitch)

    x2 = x1 * math.cos(yaw) + z1 * math.sin(yaw)
    y2 = y1
    z2 = -x1 * math.sin(yaw) + z1 * math.cos(yaw)

    x3 = x2 * math.cos(roll) - y2 * math.sin(roll)
    y3 = x2 * math.sin(roll) + y2 * math.cos(roll)
    z3 = z2

    return [x3, y3, z3]


def convert_3dbbox(point, cam_params):
    """Convert 3D bounding box to 2D image coordinates"""
    x, y, z, x_size, y_size, z_size, pitch, yaw, roll = point
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    local_corners = [
        [ hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy,  hz],
        [ hx, -hy, -hz],
        [-hx,  hy,  hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [-hx, -hy, -hz]
    ]

    img_corners = []
    for corner in local_corners:
        rotated = rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
        X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
        if Z > 0:
            x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
            y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
            img_corners.append([x_2d, y_2d])

    return img_corners


def draw_2d_bounding_boxes_on_frame(frame, bounding_boxes_2d, color=None):
    """
    Draw 2D bounding boxes on a frame.
    
    Args:
        frame: numpy array (H, W, C) in BGR format
        bounding_boxes_2d: List of dictionaries with 'bbox_2d' and 'label' fields
        color: Optional color for drawing boxes (B, G, R)
        
    Returns:
        frame: Frame with 2D bounding boxes drawn
    """
    h, w = frame.shape[:2]
    
    for i, bbox_data in enumerate(bounding_boxes_2d):
        if 'bbox_2d' in bbox_data and 'label' in bbox_data:
            # Extract 2D bounding box coordinates
            x1, y1, x2, y2 = bbox_data['bbox_2d']
            
            # Convert normalized coordinates [0, 1000] to pixel coordinates
            x1_px = int(x1 * w / 1000)
            y1_px = int(y1 * h / 1000)
            x2_px = int(x2 * w / 1000)
            y2_px = int(y2 * h / 1000)
            
            # Generate random color for each box if no color specified
            if color is None:
                box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                box_color = color
            
            # Draw rectangle
            cv2.rectangle(frame, (x1_px, y1_px), (x2_px, y2_px), box_color, 2)
            
            # Add label
            label = bbox_data.get('label', f'object_{i}')
            cv2.putText(frame, label, (x1_px, y1_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    return frame


def draw_3d_bounding_boxes_on_frame(frame, bounding_boxes, object_types, cam_params, color=None):
    """
    Draw 3D bounding boxes on a frame as wireframe cubes.
    
    Args:
        frame: numpy array (H, W, C) in BGR format
        bounding_boxes: List of 3D bounding boxes [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
        object_types: List of object type strings
        cam_params: Camera parameters dictionary with keys 'fx', 'fy', 'cx', 'cy'
        color: Optional color for drawing boxes (B, G, R)
        
    Returns:
        frame: Frame with 3D bounding boxes drawn
    """
    h, w = frame.shape[:2]
    
    # Define edges of a cube (connections between vertices)
    edges = [
        [0, 1], [2, 3], [4, 5], [6, 7],  # Front and back face edges
        [0, 2], [1, 3], [4, 6], [5, 7],  # Top and bottom edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
    ]
    
    for i, (bbox, obj_type) in enumerate(zip(bounding_boxes, object_types)):
        if len(bbox) >= 9:
            # Extract 3D bounding box parameters
            x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw = bbox
            
            # Convert angles from radians to degrees (multiply by 180 as in the original code)
            bbox_3d = [x_center, y_center, z_center, x_size, y_size, z_size, pitch * 180, yaw * 180, roll * 180]
            
            # Convert 3D bounding box to 2D image coordinates
            bbox_2d = convert_3dbbox(bbox_3d, cam_params)
            
            # Check if we have enough points to draw the box
            if len(bbox_2d) >= 8:
                # Generate random color for each box if no color specified
                if color is None:
                    box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    box_color = color
                
                # Draw edges of the 3D box
                for start, end in edges:
                    try:
                        pt1 = tuple([int(_pt) for _pt in bbox_2d[start]])
                        pt2 = tuple([int(_pt) for _pt in bbox_2d[end]])
                        cv2.line(frame, pt1, pt2, box_color, 2)
                    except:
                        continue
                
                # Add label near the first vertex
                label = f"{obj_type}"
                try:
                    label_pos = tuple([int(_pt) for _pt in bbox_2d[0]])
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                except:
                    pass
    
    return frame


def estimate_camera_parameters(width, height, fov_degrees=60):
    """
    Estimate camera parameters based on image dimensions and field of view.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Field of view in degrees (default 60)
        
    Returns:
        Dictionary with camera parameters 'fx', 'fy', 'cx', 'cy'
    """
    # Assume square pixels and symmetric camera
    cx = width / 2.0
    cy = height / 2.0
    
    # Calculate focal length from FOV
    # fov = 2 * arctan(sensor_size / (2 * focal_length))
    # For a typical sensor, we can estimate focal length
    fov_rad = np.deg2rad(fov_degrees)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_rad / 2.0)
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def apply_bounding_boxes_to_video(json_path, video_path, output_path, cam_params=None):
    """
    Apply precomputed bounding boxes to a video and save the annotated video.
    
    Args:
        json_path: Path to JSON file with bounding box data
        video_path: Path to input video file
        output_path: Path to save annotated video
        cam_params: Optional camera parameters dictionary with keys 'fx', 'fy', 'cx', 'cy'
    """
    # Read bounding box data
    with open(json_path, 'r') as f:
        bbox_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Estimate camera parameters if not provided
    if cam_params is None:
        cam_params = estimate_camera_parameters(width, height)
        print(f"Estimated camera parameters: fx={cam_params['fx']:.2f}, fy={cam_params['fy']:.2f}, cx={cam_params['cx']:.2f}, cy={cam_params['cy']:.2f}")
    else:
        print(f"Using provided camera parameters: fx={cam_params['fx']:.2f}, fy={cam_params['fy']:.2f}, cx={cam_params['cx']:.2f}, cy={cam_params['cy']:.2f}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Create a mapping from frame index to bounding box data for quick lookup
    frame_data_map = {frame_info["frame_index"]: frame_info for frame_info in bbox_data["frames"]}
    
    # Process frames
    frame_count = 0
    with tqdm(total=total_frames, desc="Applying bounding boxes") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get bounding box data for this frame if available
            if frame_count in frame_data_map:
                frame_info = frame_data_map[frame_count]
                
                # Check if we have 2D bounding boxes
                if "bounding_boxes_2d" in frame_info:
                    bounding_boxes_2d = frame_info["bounding_boxes_2d"]
                    # Draw 2D bounding boxes on frame
                    frame = draw_2d_bounding_boxes_on_frame(frame, bounding_boxes_2d)
                # Check if we have 3D bounding boxes
                elif "bounding_boxes" in frame_info and "object_types" in frame_info:
                    bounding_boxes = frame_info["bounding_boxes"]
                    object_types = frame_info["object_types"]
                    # Draw 3D bounding boxes on frame
                    frame = draw_3d_bounding_boxes_on_frame(frame, bounding_boxes, object_types, cam_params)
            
            # Write annotated frame to output video
            out.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        # Ignore errors when running in headless environments
        pass
    
    print(f"Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply precomputed bounding boxes to a video")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with bounding box data")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save annotated video")
    parser.add_argument("--fx", type=float, help="Camera focal length in x direction (pixels)")
    parser.add_argument("--fy", type=float, help="Camera focal length in y direction (pixels)")
    parser.add_argument("--cx", type=float, help="Camera principal point x coordinate (pixels)")
    parser.add_argument("--cy", type=float, help="Camera principal point y coordinate (pixels)")
    
    args = parser.parse_args()
    
    # Create camera parameters dictionary if provided
    cam_params = None
    if args.fx is not None and args.fy is not None and args.cx is not None and args.cy is not None:
        cam_params = {
            'fx': args.fx,
            'fy': args.fy,
            'cx': args.cx,
            'cy': args.cy
        }
    
    # Apply bounding boxes to video
    apply_bounding_boxes_to_video(
        json_path=args.json_path,
        video_path=args.video_path,
        output_path=args.output_path,
        cam_params=cam_params
    )
    
    print("Bounding boxes applied to video successfully!")


if __name__ == "__main__":
    main()