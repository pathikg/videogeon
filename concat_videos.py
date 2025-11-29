"""Concatenate all scene videos into a single video file."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import config


def find_scene_videos(output_dir: Path) -> List[Path]:
    """
    Find all scene video files in the output directory, sorted by scene number.

    Args:
        output_dir: Directory containing scene videos

    Returns:
        List of video file paths in order (scene_1, scene_2, etc.)
    """
    scene_videos = []
    scene_num = 1

    while True:
        video_path = output_dir / f"scene_{scene_num}.mp4"
        if video_path.exists():
            scene_videos.append(video_path)
            scene_num += 1
        else:
            break

    return scene_videos


def create_concat_file(video_paths: List[Path], concat_file: Path) -> None:
    """
    Create a concat file for ffmpeg.

    Args:
        video_paths: List of video file paths
        concat_file: Path to write the concat file
    """
    with open(concat_file, "w") as f:
        for video_path in video_paths:
            # Use absolute path to avoid path resolution issues
            abs_path = video_path.resolve()
            # Escape single quotes and special characters for ffmpeg
            escaped_path = str(abs_path).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")


def concatenate_videos(
    output_dir: Path,
    output_filename: str = "final_video.mp4",
    video_paths: Optional[List[Path]] = None,
) -> Path:
    """
    Concatenate all scene videos into a single video file using ffmpeg.

    Args:
        output_dir: Directory containing scene videos
        output_filename: Name for the output concatenated video
        video_paths: Optional list of video paths (if None, will find automatically)

    Returns:
        Path to the concatenated video file
    """
    if video_paths is None:
        video_paths = find_scene_videos(output_dir)

    if not video_paths:
        raise ValueError(f"No scene videos found in {output_dir}")

    print(f"Found {len(video_paths)} scene videos:")
    for i, video_path in enumerate(video_paths, 1):
        print(f"  {i}. {video_path.name}")

    # Create temporary concat file
    concat_file = output_dir / "concat_list.txt"

    try:
        create_concat_file(video_paths, concat_file)

        # Output path
        output_path = output_dir / output_filename

        print(f"\nConcatenating videos...")
        print(f"Output: {output_path}")

        # Run ffmpeg to concatenate videos
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",  # Copy streams without re-encoding (faster)
            "-y",  # Overwrite output file if it exists
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"\n✓ Successfully created: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error concatenating videos:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        raise

    except FileNotFoundError:
        print("\n✗ Error: ffmpeg not found. Please install ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
        raise

    finally:
        # Clean up concat file
        if concat_file.exists():
            concat_file.unlink()


def find_latest_output_dir() -> Optional[Path]:
    """
    Find the most recent output directory.

    Returns:
        Path to the latest output directory, or None if not found
    """
    output_base = Path(config.OUTPUT_DIR)
    if not output_base.exists():
        return None

    # Find all timestamped directories
    dirs = [d for d in output_base.iterdir() if d.is_dir()]

    if not dirs:
        return None

    # Sort by modification time, most recent first
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]


def main():
    """Main function to concatenate videos."""
    if len(sys.argv) > 1:
        # Use provided directory
        output_dir = Path(sys.argv[1])
        if not output_dir.exists():
            print(f"Error: Directory not found: {output_dir}")
            sys.exit(1)
    else:
        # Find latest output directory
        output_dir = find_latest_output_dir()
        if output_dir is None:
            print(f"Error: No output directories found in {config.OUTPUT_DIR}")
            print(f"\nUsage: python concat_videos.py [output_directory]")
            sys.exit(1)
        else:
            print(f"Using latest output directory: {output_dir}")

    try:
        output_path = concatenate_videos(output_dir)
        print(f"\n✓ Final video saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to concatenate videos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
