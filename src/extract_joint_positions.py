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

def extract_actions_from_inference(input_file, output_file):
    """
    Read inference results JSON file and extract the first action from each result.
    
    Args:
        input_file (str): Path to the input inference results JSON file
        output_file (str): Path to the output JSON file with actions array
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        inference_results = json.load(f)
    
    # Extract the first action from each result
    actions_array = []
    for item in inference_results:
        if 'result' in item and 'action' in item['result']:
            # Get the first action from the action array
            if item['result']['action'] and len(item['result']['action']) > 0:
                actions_array.append(item['result']['action'][0])
    
    # Save the array of actions to the output file
    with open(output_file, 'w') as f:
        json.dump(actions_array, f, indent=2)
    
    print(f"Extracted {len(actions_array)} actions from inference results")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Define input and output file paths for joint positions
    input_file = "temp/metadata_20251113_080958.json"
    output_file = "temp/metadata_20251113_080958_gt.json"
    
    # Extract joint positions
    extract_joint_positions(input_file, output_file)
    
    # Define input and output file paths for actions
    inference_input_file = "temp/inference_results.json"
    inference_output_file = "temp/inference_actions.json"
    
    # Extract actions from inference results
    extract_actions_from_inference(inference_input_file, inference_output_file)
