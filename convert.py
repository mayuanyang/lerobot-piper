#!/usr/bin/env python3
"""
Script to convert joint positions in JSON files by dividing by a factor.
This script processes all JSON files in the data/piper_training_data/episode*/ directories
and converts the joint_positions values by dividing them by the specified factor.
"""

import os
import json
import glob

# Conversion factor
FACTOR = 57324.840764

def process_json_file(file_path):
    """
    Process a single JSON file and convert joint positions.
    
    Args:
        file_path (str): Path to the JSON file
    """
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each frame's joint_positions
        for frame in data.get('frames', []):
            if 'joint_positions' in frame:
                # Divide each joint position by the factor
                frame['joint_positions'] = [pos / FACTOR for pos in frame['joint_positions']]
        
        # Write the modified data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_all_files():
    """
    Process all JSON files in the data directory.
    """
    # Find all JSON files in the episode directories
    pattern = os.path.join('src', 'data', 'piper_training_data', 'episode*', '*.json')
    json_files = glob.glob(pattern)
    
    if not json_files:
        # Try alternative path pattern
        pattern = os.path.join('data', 'piper_training_data', 'episode*', '*.json')
        json_files = glob.glob(pattern)
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for file_path in json_files:
        process_json_file(file_path)
    
    print("All files processed!")

# Main execution
if __name__ == "__main__":
    process_all_files()
