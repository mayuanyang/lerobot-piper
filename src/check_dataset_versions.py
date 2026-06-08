"""
Script to check LeRobot dataset versions in the ISdept/community_dataset_v3_part1 repository.
Loops through each sub-dataset (user/demo structure) and checks the version (v2 or v3) 
by reading meta/info.json. Generates a markdown file with the results.
"""

import json
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import time


def is_folder(item) -> bool:
    """Check if an item is a folder."""
    class_name = type(item).__name__
    return 'Folder' in class_name


def get_user_folders(repo_id: str) -> list[str]:
    """Get list of user folders in the repository."""
    api = HfApi()
    tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo="")
    
    folders = []
    for item in tree:
        if is_folder(item) and not item.path.startswith('.'):
            folders.append(item.path)
    
    return sorted(folders)


def get_demo_folders(repo_id: str, user_folder: str) -> list[str]:
    """Get list of demo folders within a user folder."""
    api = HfApi()
    tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=user_folder)
    
    folders = []
    for item in tree:
        if is_folder(item):
            folders.append(item.path)
    
    return sorted(folders)


def check_dataset_version(repo_id: str, dataset_path: str) -> dict:
    """
    Check the LeRobot dataset version for a dataset.
    Returns dict with version info.
    """
    result = {
        "path": dataset_path,
        "version": "unknown",
        "total_episodes": "N/A",
        "total_frames": "N/A",
        "fps": "N/A",
        "cameras": [],
        "error": None
    }
    
    try:
        info_path = f"{dataset_path}/meta/info.json"
        
        # Download and read info.json
        info_file = hf_hub_download(
            repo_id=repo_id,
            filename=info_path,
            repo_type="dataset"
        )
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # Extract version info
        result["version"] = info.get("codebase_version", "unknown")
        result["total_episodes"] = info.get("total_episodes", "N/A")
        result["total_frames"] = info.get("total_frames", "N/A")
        result["fps"] = info.get("fps", "N/A")
        
        # Check for cameras
        features = info.get("features", {})
        cameras = [k for k in features.keys() if k.startswith("observation.images.")]
        result["cameras"] = cameras
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def generate_markdown(all_results: list[dict], output_path: Path):
    """Generate a markdown file with dataset version information."""
    
    # Count versions and aggregate stats
    valid_results = [r for r in all_results if r["error"] is None]
    v2_results = [r for r in valid_results if r["version"].startswith("v2")]
    v3_results = [r for r in valid_results if r["version"].startswith("v3")]
    unknown_results = [r for r in valid_results if not r["version"].startswith(("v2", "v3"))]
    
    v2_count = len(v2_results)
    v3_count = len(v3_results)
    unknown_count = len(unknown_results)
    error_count = sum(1 for r in all_results if r["error"] is not None)
    
    # Calculate total episodes and frames for v2 and v3
    v2_total_episodes = sum(r["total_episodes"] for r in v2_results if isinstance(r["total_episodes"], int))
    v2_total_frames = sum(r["total_frames"] for r in v2_results if isinstance(r["total_frames"], int))
    v3_total_episodes = sum(r["total_episodes"] for r in v3_results if isinstance(r["total_episodes"], int))
    v3_total_frames = sum(r["total_frames"] for r in v3_results if isinstance(r["total_frames"], int))
    
    total_episodes = v2_total_episodes + v3_total_episodes
    total_frames = v2_total_frames + v3_total_frames
    
    md = []
    md.append("# ISdept/community_dataset_v3_part1 - Dataset Version Report\n")
    md.append(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append("## Summary\n")
    md.append(f"- **Total datasets (demos):** {len(all_results)}")
    md.append(f"- **Total episodes:** {total_episodes:,}")
    md.append(f"- **Total frames:** {total_frames:,}")
    md.append(f"- **Errors/Inaccessible:** {error_count}\n")
    
    md.append("### Version Breakdown\n")
    md.append("| Version | # Datasets | Total Episodes | Total Frames |")
    md.append("|---------|------------|----------------|--------------|")
    md.append(f"| **v3.x** | {v3_count} | {v3_total_episodes:,} | {v3_total_frames:,} |")
    md.append(f"| **v2.x** | {v2_count} | {v2_total_episodes:,} | {v2_total_frames:,} |")
    if unknown_count > 0:
        md.append(f"| **Unknown** | {unknown_count} | N/A | N/A |")
    md.append("")
    
    # Group by user
    user_groups = {}
    for r in all_results:
        user = r["path"].split("/")[0]
        if user not in user_groups:
            user_groups[user] = []
        user_groups[user].append(r)
    
    # Detailed table by user
    md.append("## Detailed Dataset Information by User\n")
    
    for user in sorted(user_groups.keys()):
        demos = user_groups[user]
        md.append(f"\n### {user} ({len(demos)} demos)\n")
        md.append("| # | Demo Name | Version | Episodes | Frames | FPS | Cameras | Status |")
        md.append("|---|-----------|---------|----------|--------|-----|---------|--------|")
        
        for i, r in enumerate(sorted(demos, key=lambda x: x["path"]), 1):
            demo_name = r["path"].split("/")[-1]
            version = r["version"]
            episodes = r["total_episodes"]
            frames = r["total_frames"]
            fps = r["fps"]
            
            # Camera info
            cameras = r.get("cameras", [])
            if cameras:
                cam_names = [c.replace("observation.images.", "") for c in cameras]
                cam_str = ", ".join(cam_names)
            else:
                cam_str = "N/A"
            
            # Status
            if r["error"]:
                status = f"⚠️ Error"
            elif version.startswith("v3"):
                status = "✅ v3"
            elif version.startswith("v2"):
                status = "✅ v2"
            else:
                status = "❓ Unknown"
            
            md.append(f"| {i} | {demo_name} | {version} | {episodes} | {frames} | {fps} | {cam_str} | {status} |")
    
    # Version breakdown sections
    md.append("\n---\n")
    md.append("## LeRobot v3.x Datasets\n")
    v3_datasets = [r for r in all_results if r["version"].startswith("v3") and r["error"] is None]
    if v3_datasets:
        for r in sorted(v3_datasets, key=lambda x: x["path"]):
            md.append(f"- **{r['path']}**: {r['total_episodes']} episodes, {r['total_frames']} frames")
    else:
        md.append("*No v3.x datasets found.*")
    
    md.append("\n## LeRobot v2.x Datasets\n")
    v2_datasets = [r for r in all_results if r["version"].startswith("v2") and r["error"] is None]
    if v2_datasets:
        for r in sorted(v2_datasets, key=lambda x: x["path"]):
            md.append(f"- **{r['path']}**: {r['total_episodes']} episodes, {r['total_frames']} frames")
    else:
        md.append("*No v2.x datasets found.*")
    
    md.append("\n## Datasets with Issues\n")
    error_datasets = [r for r in all_results if r["error"]]
    if error_datasets:
        for r in sorted(error_datasets, key=lambda x: x["path"]):
            md.append(f"- **{r['path']}**: {r['error'][:100]}...")
    else:
        md.append("*All datasets processed successfully.*")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(md))
    
    print(f"\n✅ Markdown report saved to: {output_path}")


def main():
    repo_id = "ISdept/community_dataset_v3_part1"
    output_path = Path("notes/community_dataset_version_report.md")
    
    print(f"🔍 Checking dataset versions in: {repo_id}")
    print("=" * 60)
    
    # Get all user folders
    print("\n📁 Discovering user folders...")
    user_folders = get_user_folders(repo_id)
    print(f"Found {len(user_folders)} user folders")
    
    # Check each user's demos
    all_results = []
    total_demos = 0
    
    for u_idx, user in enumerate(user_folders, 1):
        print(f"\n[{u_idx}/{len(user_folders)}] Scanning user: {user}")
        
        try:
            demo_folders = get_demo_folders(repo_id, user)
            print(f"  Found {len(demo_folders)} demos")
            total_demos += len(demo_folders)
            
            for d_idx, demo in enumerate(demo_folders, 1):
                demo_name = demo.split("/")[-1]
                print(f"  [{d_idx}/{len(demo_folders)}] Checking: {demo_name}")
                
                result = check_dataset_version(repo_id, demo)
                all_results.append(result)
                
                if result["error"]:
                    print(f"    ⚠️ Error: {result['error'][:80]}...")
                else:
                    print(f"    ✅ Version: {result['version']}, Episodes: {result['total_episodes']}, Frames: {result['total_frames']}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.3)
                
        except Exception as e:
            print(f"  ❌ Error scanning user {user}: {e}")
    
    # Generate markdown report
    print("\n" + "=" * 60)
    print(f"📊 Generating markdown report... (Total: {len(all_results)} datasets)")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown(all_results, output_path)
    
    # Print summary
    valid_results = [r for r in all_results if r["error"] is None]
    v2_count = sum(1 for r in valid_results if r["version"].startswith("v2"))
    v3_count = sum(1 for r in valid_results if r["version"].startswith("v3"))
    unknown_count = sum(1 for r in valid_results if not r["version"].startswith(("v2", "v3")))
    error_count = sum(1 for r in all_results if r["error"] is not None)
    
    print("\n📋 Summary:")
    print(f"  Total datasets: {len(all_results)}")
    print(f"  v2.x: {v2_count}")
    print(f"  v3.x: {v3_count}")
    print(f"  Unknown: {unknown_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    main()