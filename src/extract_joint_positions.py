import json

def extract_joint_positions(input_file, output_file):
    """
    Read metadata JSON file and extract joint_positions from each frame.
    
    Args:
        input_file (str): Path to the input metadata JSON file
        output_file (str): Path to the output JSON file with joint positions array
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract joint_positions from each frame
    joint_positions_array = []
    for frame in metadata.get('frames', []):
        if 'joint_positions' in frame:
            joint_positions_array.append(frame['joint_positions'])
    
    # Save the array of joint_positions to the output file
    with open(output_file, 'w') as f:
        json.dump(joint_positions_array, f, indent=2)
    
    print(f"Extracted {len(joint_positions_array)} frames of joint positions")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "temp/metadata_20251113_080958.json"
    output_file = "temp/metadata_20251113_080958_gt.json"
    
    # Extract joint positions
    extract_joint_positions(input_file, output_file)
