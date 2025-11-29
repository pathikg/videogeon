"""Main video generation pipeline using Replicate."""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from google import genai
from tqdm import tqdm

import config
from story_generator import generate_story, generate_frame_spec, generate_video_prompt
from video_generator import (
    generate_initial_image,
    generate_first_video_replicate,
    generate_video_replicate,
    extract_last_frame,
)

from dotenv import load_dotenv

load_dotenv()


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def create_output_dir() -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clamp_replicate_duration(duration: int) -> int:
    """
    Clamp duration to valid Replicate values (5 or 10 seconds).

    Args:
        duration: Desired duration in seconds

    Returns:
        Clamped duration (5 or 10)
    """
    # Replicate only accepts 5 or 10 seconds
    # Round to nearest: <=7 -> 5, >7 -> 10
    # if duration <= 7:
    #     return 5
    # else:
    #     return 10

    return 5  # need to save money for my children's future


def clamp_replicate_aspect_ratio(aspect_ratio: str) -> str:
    """
    Clamp aspect ratio to valid Replicate values ("16:9", "9:16", "1:1").

    Args:
        aspect_ratio: Desired aspect ratio string

    Returns:
        Clamped aspect ratio ("16:9", "9:16", or "1:1")
    """
    # Replicate only accepts "16:9", "9:16", or "1:1"
    valid_ratios = ["16:9", "9:16", "1:1"]

    # If already valid, return as-is
    if aspect_ratio in valid_ratios:
        return aspect_ratio

    # Try to parse and map to nearest valid ratio
    try:
        # Parse ratio like "16:9" or "4:3"
        parts = aspect_ratio.split(":")
        if len(parts) == 2:
            width = float(parts[0])
            height = float(parts[1])
            ratio = width / height

            # Map to nearest valid ratio
            if ratio > 1.5:  # Wider than 16:9 (1.78)
                return "16:9"
            elif ratio < 0.6:  # Taller than 9:16 (0.56)
                return "9:16"
            else:  # Close to square
                return "1:1"
    except (ValueError, IndexError):
        pass

    # Default to 16:9 if parsing fails
    return "16:9"


def run_pipeline_replicate(
    user_prompt: str,
    save_intermediates: bool = True,
    initial_image_path: Optional[str] = None,
) -> Path:
    """
    Run the full video generation pipeline using Replicate.

    Args:
        user_prompt: User's story idea
        save_intermediates: Whether to save intermediate files (story, prompts, etc.)
        initial_image_path: Optional path to an existing image file to use as the first frame.
                           If provided, skips image generation and uses this image instead.

    Returns:
        Path to output directory containing all generated files
    """
    client = genai.Client()
    output_dir = create_output_dir()

    print("=" * 60)
    print("VIDEO GENERATION PIPELINE (REPLICATE)")
    print("=" * 60)

    pipeline_start = time.time()

    # Step 1: Generate story breakdown
    print("\n[1/5] Generating story breakdown...")
    step_start = time.time()
    story_data = generate_story(user_prompt, client)
    step_time = time.time() - step_start

    if save_intermediates:
        story_path = output_dir / "story.json"
        story_path.write_text(json.dumps(story_data, indent=2, ensure_ascii=False))
        print(f"      Saved to {story_path}")

    print(f"      Title: {story_data['global_context']['title']}")
    print(f"      Genre: {story_data['global_context']['genre']}")
    print(f"      Time taken: {format_time(step_time)}")

    # Step 2: Generate frame specification for Scene 1
    print("\n[2/5] Generating initial frame specification...")
    step_start = time.time()
    frame_spec = generate_frame_spec(story_data, scene_number=1, client=client)
    step_time = time.time() - step_start

    if save_intermediates:
        frame_path = output_dir / "frame_spec.json"
        frame_path.write_text(json.dumps(frame_spec, indent=2, ensure_ascii=False))
        print(f"      Saved to {frame_path}")

    print(f"      Time taken: {format_time(step_time)}")

    # Step 3: Generate video prompts for all scenes
    print("\n[3/5] Generating video prompts...")
    step_start = time.time()
    video_prompts = []

    for scene_num in tqdm(range(1, config.NUM_SCENES + 1), desc="Prompts"):
        scene = next(s for s in story_data["scenes"] if s["scene_number"] == scene_num)

        if scene_num == 1:
            prompt_data = generate_video_prompt(
                frame_spec=frame_spec, scene_data=scene, client=client
            )
        else:
            prompt_data = generate_video_prompt(scene_data=scene, client=client)

        video_prompts.append(prompt_data)

    step_time = time.time() - step_start

    if save_intermediates:
        prompts_path = output_dir / "video_prompts.json"
        prompts_path.write_text(json.dumps(video_prompts, indent=2, ensure_ascii=False))
        print(f"      Saved to {prompts_path}")

    print(f"      Time taken: {format_time(step_time)}")

    # Step 4: Generate or load initial image for Scene 1
    image_path = output_dir / "initial_frame.png"
    if initial_image_path:
        print("\n[4/5] Loading initial frame image from file...")
        step_start = time.time()
        # Copy the user's image to output directory
        from shutil import copy2

        copy2(initial_image_path, image_path)
        step_time = time.time() - step_start
        print(f"      Loaded from {initial_image_path}")
        print(f"      Copied to {image_path}")
        print(f"      Time taken: {format_time(step_time)}")
    else:
        print("\n[4/5] Generating initial frame image...")
        step_start = time.time()
        image_prompt = video_prompts[0]["video_prompt"]
        initial_image = generate_initial_image(
            image_prompt, client, save_path=str(image_path)
        )
        step_time = time.time() - step_start
        print(f"      Saved to {image_path}")
        print(f"      Time taken: {format_time(step_time)}")

    # Step 5: Generate videos using Replicate
    print("\n[5/5] Generating videos with Replicate...")
    step_start = time.time()
    video_paths = []

    # Get aspect ratio and duration from config or video prompts
    aspect_ratio = (
        video_prompts[0]
        .get("parameters", {})
        .get("aspect_ratio", config.REPLICATE_ASPECT_RATIO)
    )
    duration = (
        video_prompts[0]
        .get("parameters", {})
        .get("duration", config.REPLICATE_DURATION)
    )
    # Clamp to valid Replicate values
    aspect_ratio = clamp_replicate_aspect_ratio(aspect_ratio)
    duration = clamp_replicate_duration(duration)

    # Scene 1: Image-to-Video
    print("      Scene 1 (Image-to-Video)...")
    scene_start = time.time()
    video_1_path = output_dir / "scene_1.mp4"
    negative_prompt_1 = video_prompts[0].get("negative_prompt", "")

    generate_first_video_replicate(
        prompt=video_prompts[0]["video_prompt"],
        image_path=str(image_path),
        output_path=str(video_1_path),
        duration=duration,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt_1,
    )
    video_paths.append(video_1_path)
    scene_time = time.time() - scene_start
    print(f"      Saved to {video_1_path}")
    print(f"      Time taken: {format_time(scene_time)}")

    # Scenes 2-4: Extract last frame and generate next video
    current_video_path = video_1_path

    for scene_num in tqdm(range(2, config.NUM_SCENES + 1), desc="Scenes"):
        print(f"      Scene {scene_num} (Extract frame & Generate)...")
        scene_start = time.time()

        # Extract last frame from previous video
        last_frame_path = output_dir / f"scene_{scene_num - 1}_last_frame.png"
        extract_last_frame(str(current_video_path), str(last_frame_path))
        print(f"      Extracted last frame to {last_frame_path}")

        # Generate next video using the extracted frame
        video_path = output_dir / f"scene_{scene_num}.mp4"
        negative_prompt = video_prompts[scene_num - 1].get("negative_prompt", "")

        # Get aspect ratio and duration for this scene
        scene_aspect_ratio = (
            video_prompts[scene_num - 1]
            .get("parameters", {})
            .get("aspect_ratio", aspect_ratio)
        )
        scene_duration = (
            video_prompts[scene_num - 1].get("parameters", {}).get("duration", duration)
        )
        # Clamp to valid Replicate values
        scene_aspect_ratio = clamp_replicate_aspect_ratio(scene_aspect_ratio)
        scene_duration = clamp_replicate_duration(scene_duration)

        generate_video_replicate(
            prompt=video_prompts[scene_num - 1]["video_prompt"],
            start_image_path=str(last_frame_path),
            output_path=str(video_path),
            duration=scene_duration,
            aspect_ratio=scene_aspect_ratio,
            negative_prompt=negative_prompt,
        )
        video_paths.append(video_path)
        current_video_path = video_path

        scene_time = time.time() - scene_start
        print(f"      Saved to {video_path}")
        print(f"      Time taken: {format_time(scene_time)}")

    step_time = time.time() - step_start
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Output directory: {output_dir}")
    print(
        f"Generated {len(video_paths)} video segments ({config.REPLICATE_DURATION * len(video_paths)} seconds total)"
    )
    print(f"Total time: {format_time(total_time)}")
    print("=" * 60)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate video from story prompt using Replicate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_replicate.py "A lonely astronaut discovers a flower on Mars"
  python pipeline_replicate.py "A lonely astronaut discovers a flower on Mars" --image path/to/image.png
        """,
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Story prompt for video generation",
    )
    parser.add_argument(
        "--image",
        type=str,
        dest="initial_image_path",
        help="Path to an existing image file to use as the first frame (skips image generation)",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_false",
        dest="save_intermediates",
        help="Don't save intermediate files (story.json, prompts, etc.)",
    )

    args = parser.parse_args()

    run_pipeline_replicate(
        user_prompt=args.prompt,
        save_intermediates=args.save_intermediates,
        initial_image_path=args.initial_image_path,
    )
