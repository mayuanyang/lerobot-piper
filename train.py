from pathlib import Path

import torch
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
    training_steps = 5000
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
    fps = 27.38
    delta_timestamps = {
        # Load the previous image and state at ~-0.1095 seconds (which is -3 * 1/fps) before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-3/fps, 0.0],
        "observation.state": [-3/fps, 0.0],
        # Load the previous action (~-0.1095s), the next action to be executed (0.0),
        # and 14 future actions with proper spacing as multiples of 1/fps. All these actions will be
        # used to supervise the policy.
        "action": [-3/fps, 0.0, 1/fps, 2/fps, 3/fps, 4/fps, 5/fps, 6/fps, 7/fps, 8/fps, 9/fps, 10/fps, 11/fps, 12/fps, 13/fps, 14/fps],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, force_cache_sync=True, revision="main")

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
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
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
