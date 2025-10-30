from pathlib import Path

import torch
from tqdm import tqdm
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        # Return previous 1 frame, itself, and next 13 frames
        return [i / fps for i in range(-1, 14)]  # -1, 0, 1, 2, ..., 13

    return [i / fps for i in delta_indices]


from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def train(output_dir, dataset_id="ISdept/piper_arm", push_to_hub=False):
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

        # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 100
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(dataset_id, force_cache_sync=True, revision="main")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

 
    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    # Using multiples of 1/27.38 ≈ 0.0365 to match the dataset FPS
    # Note: observation.image and observation.state are handled separately by LeRobot and not included in the parquet file
    # Calculating exact multiples of 1/27.38 for delta timestamps
    fps = 10
    frame_time = 1 / fps  # 0.1 seconds

    delta_timestamps = {
        # 1. Observation (Image and State)
        # The policy needs the current frame (0.0) and context from previous frames.
        # The original example used -3 * 1/fps. For FPS=10, this is -3 * 0.1 = -0.3 seconds.
        "observation.image": [-3 * frame_time, 0.0],  # -0.3 seconds (3 frames back) and current frame (0.0)
        "observation.state": [-3 * frame_time, 0.0],  # -0.3 seconds (3 frames back) and current frame (0.0)

        # 2. Action (Future Prediction Horizon)
        # The policy predicts a sequence of future actions.
        # The sequence should start at the *next* action to be executed (often 0.0 or 1 * frame_time)
        # and cover a prediction horizon (16 actions total in your example).
        
        # Your old action array had 16 values. We'll match that length:
        # Action at -3/fps, then action at 0.0, then 14 future actions (16 total).
        "action": [
            -3 * frame_time,  # Action from 3 frames ago (-0.3s)
            0.0 * frame_time, # Current action/action to execute now (0.0s)
            1 * frame_time,   # Action for 1 frame ahead (+0.1s)
            2 * frame_time,   # Action for 2 frames ahead (+0.2s)
            3 * frame_time,
            4 * frame_time,
            5 * frame_time,
            6 * frame_time,
            7 * frame_time,
            8 * frame_time,
            9 * frame_time,
            10 * frame_time,
            11 * frame_time,
            12 * frame_time,
            13 * frame_time,
            14 * frame_time,  # Action for 14 frames ahead (+1.4s)
        ]
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main", tolerance_s=0.01)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Step {step}")
        for batch_idx, batch in prog_bar:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                prog_bar.set_postfix({"step": step, "loss": f"{loss.item():.3f}"})
            step += 1
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    if push_to_hub:
        repo = 'ISdept/piper_arm_diffusion'
        policy.push_to_hub(repo)
        preprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)
        postprocessor.push_to_hub(repo)
