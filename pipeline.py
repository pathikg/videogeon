"""Main video generation pipeline."""

import base64
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types
from tqdm import tqdm

import config
from story_generator import generate_story, generate_frame_spec, generate_video_prompt
from video_generator import (
    generate_initial_image,
    load_image_from_file,
    generate_first_video,
    extend_video,
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


def run_pipeline(
    user_prompt: str,
    save_intermediates: bool = True,
    initial_image_path: Optional[str] = None,
    asset_dir: Optional[str] = None,
    reuse_output_dir: Optional[str] = None,
) -> Path:
    """
    Run the full video generation pipeline.

    Args:
        user_prompt: User's story idea (ignored if reuse_output_dir is provided)
        save_intermediates: Whether to save intermediate files (story, prompts, etc.)
        initial_image_path: Optional path to an existing image file to use as the first frame.
                           If provided, skips image generation and uses this image instead.
        asset_dir: Optional path to a directory containing image files to use as reference assets.
        reuse_output_dir: Optional path to existing output directory to reuse (skips story/prompt generation).

    Returns:
        Path to output directory containing all generated files
    """
    client = genai.Client()

    # Use existing output directory if provided, otherwise create new one
    if reuse_output_dir:
        output_dir = Path(reuse_output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")
        print(f"Reusing existing output directory: {output_dir}")
    else:
        output_dir = create_output_dir()

    # Load reference images from directory if provided
    reference_images = None
    if asset_dir:
        print("\n[0/5] Loading reference assets from directory...")
        asset_dir_path = Path(asset_dir)
        if not asset_dir_path.exists() or not asset_dir_path.is_dir():
            raise FileNotFoundError(f"Asset directory not found: {asset_dir_path}")

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

        # Find all image files in the directory
        asset_files = [
            f
            for f in asset_dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not asset_files:
            raise ValueError(
                f"No image files found in asset directory: {asset_dir_path}"
            )

        reference_images = []
        for asset_path in sorted(asset_files):
            # Load image using Image.from_file()
            asset_image = types.Image.from_file(location=str(asset_path))

            reference_image = types.VideoGenerationReferenceImage(
                image=asset_image,
                reference_type=types.VideoGenerationReferenceType.ASSET,
            )
            reference_images.append(reference_image)
        print(
            f"      Loaded {len(reference_images)} reference asset(s) from {asset_dir_path}"
        )

    print("=" * 60)
    print("VIDEO GENERATION PIPELINE")
    print("=" * 60)

    pipeline_start = time.time()

    # Load or generate story, frame spec, and video prompts
    if reuse_output_dir:
        # Load existing files
        print("\n[1-3/5] Loading existing story, frame spec, and video prompts...")
        step_start = time.time()

        story_path = output_dir / "story.json"
        if not story_path.exists():
            raise FileNotFoundError(f"story.json not found in {output_dir}")
        story_data = json.loads(story_path.read_text())
        print(f"      Loaded story: {story_data['global_context']['title']}")

        frame_spec_path = output_dir / "frame_spec.json"
        frame_spec = None
        if frame_spec_path.exists():
            frame_spec = json.loads(frame_spec_path.read_text())
            print(f"      Loaded frame_spec.json")

        prompts_path = output_dir / "video_prompts.json"
        if not prompts_path.exists():
            raise FileNotFoundError(f"video_prompts.json not found in {output_dir}")
        video_prompts = json.loads(prompts_path.read_text())
        print(f"      Loaded {len(video_prompts)} video prompt(s)")

        step_time = time.time() - step_start
        print(f"      Time taken: {format_time(step_time)}")
    else:
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
            scene = next(
                s for s in story_data["scenes"] if s["scene_number"] == scene_num
            )

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
            prompts_path.write_text(
                json.dumps(video_prompts, indent=2, ensure_ascii=False)
            )
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
        initial_image = load_image_from_file(str(image_path), client)
        step_time = time.time() - step_start
        print(f"      Loaded from {initial_image_path}")
        print(f"      Copied to {image_path}")
        print(f"      Time taken: {format_time(step_time)}")
    elif reuse_output_dir and image_path.exists():
        print("\n[4/5] Loading existing initial frame image...")
        step_start = time.time()
        initial_image = load_image_from_file(str(image_path), client)
        step_time = time.time() - step_start
        print(f"      Loaded from {image_path}")
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

    # Step 5: Generate videos
    print("\n[5/5] Generating videos...")
    step_start = time.time()
    videos = []

    # Scene 1: Image-to-Video
    print("      Scene 1 (Image-to-Video)...")
    scene_start = time.time()
    video_1_path = output_dir / "scene_1.mp4"
    video_1 = generate_first_video(
        prompt=video_prompts[0]["video_prompt"],
        image=initial_image,
        client=client,
        output_path=str(video_1_path),
        reference_images=reference_images,
    )
    videos.append(video_1)
    scene_time = time.time() - scene_start
    print(f"      Saved to {video_1_path}")
    print(f"      Time taken: {format_time(scene_time)}")

    # Scenes 2-4: Video Extension
    current_video = video_1

    for scene_num in range(2, config.NUM_SCENES + 1):
        print(f"      Scene {scene_num} (Video Extension)...")
        scene_start = time.time()
        video_path = output_dir / f"scene_{scene_num}.mp4"

        current_video = extend_video(
            prompt=video_prompts[scene_num - 1]["video_prompt"],
            previous_video=current_video,
            client=client,
            output_path=str(video_path),
            scene_number=scene_num,
            reference_images=reference_images,
        )
        videos.append(current_video)
        scene_time = time.time() - scene_start
        print(f"      Saved to {video_path}")
        print(f"      Time taken: {format_time(scene_time)}")

    step_time = time.time() - step_start
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Output directory: {output_dir}")
    print(
        f"Generated {len(videos)} video segments ({config.VIDEO_DURATION * len(videos)} seconds total)"
    )
    print(f"Total time: {format_time(total_time)}")
    print("=" * 60)

    return output_dir


def run_pipeline_simple(
    user_prompt: str, initial_image_path: Optional[str] = None
) -> Path:
    """
    Simplified pipeline that generates videos without intermediate LLM calls.
    Uses scene descriptions directly as video prompts.

    Args:
        user_prompt: User's story idea
        initial_image_path: Optional path to an existing image file to use as the first frame.
                           If provided, skips image generation and uses this image instead.

    Returns:
        Path to output directory
    """
    client = genai.Client()
    output_dir = create_output_dir()

    print("=" * 60)
    print("VIDEO GENERATION PIPELINE (SIMPLE)")
    print("=" * 60)

    pipeline_start = time.time()

    # Step 1: Generate story
    print("\n[1/3] Generating story...")
    step_start = time.time()
    story_data = generate_story(user_prompt, client)
    step_time = time.time() - step_start

    story_path = output_dir / "story.json"
    story_path.write_text(json.dumps(story_data, indent=2, ensure_ascii=False))
    print(f"      Title: {story_data['global_context']['title']}")
    print(f"      Time taken: {format_time(step_time)}")

    # Step 2: Generate or load initial image
    if initial_image_path:
        print("\n[2/3] Loading initial frame image from file...")
        step_start = time.time()
        image_path = output_dir / "initial_frame.png"
        # Copy the user's image to output directory
        from shutil import copy2

        copy2(initial_image_path, image_path)
        initial_image = load_image_from_file(str(image_path), client)
        step_time = time.time() - step_start
        print(f"      Loaded from {initial_image_path}")
        print(f"      Copied to {image_path}")
        print(f"      Time taken: {format_time(step_time)}")
    else:
        print("\n[2/3] Generating initial image...")
        step_start = time.time()
        scene_1 = story_data["scenes"][0]
        image_path = output_dir / "initial_frame.png"
        initial_image = generate_initial_image(
            scene_1["description"], client, save_path=str(image_path)
        )
        step_time = time.time() - step_start
        print(f"      Saved to {image_path}")
        print(f"      Time taken: {format_time(step_time)}")

    # Step 3: Generate videos using scene descriptions directly
    print("\n[3/3] Generating videos...")
    step_start = time.time()

    # Scene 1
    scene_start = time.time()
    video_1_path = output_dir / "scene_1.mp4"
    current_video = generate_first_video(
        prompt=scene_1["description"] + " " + scene_1["action"],
        image=initial_image,
        client=client,
        output_path=str(video_1_path),
    )
    scene_time = time.time() - scene_start
    print(f"      Scene 1 saved to {video_1_path}")
    print(f"      Time taken: {format_time(scene_time)}")

    # Scenes 2-4
    for i, scene in enumerate(story_data["scenes"][1:], start=2):
        scene_start = time.time()
        video_path = output_dir / f"scene_{i}.mp4"
        current_video = extend_video(
            prompt=scene["description"] + " " + scene["action"],
            previous_video=current_video,
            client=client,
            output_path=str(video_path),
            scene_number=i,
        )
        scene_time = time.time() - scene_start
        print(f"      Scene {i} saved to {video_path}")
        print(f"      Time taken: {format_time(scene_time)}")

    step_time = time.time() - step_start
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 60)
    print(f"COMPLETE - Output: {output_dir}")
    print(f"Total time: {format_time(total_time)}")
    print("=" * 60)

    return output_dir


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate video from story prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py "A lonely astronaut discovers a flower on Mars"
  python pipeline.py "A lonely astronaut discovers a flower on Mars" --image path/to/image.png
  python pipeline.py "Story prompt" --assets path/to/assets/folder
  python pipeline.py "Story prompt" --reuse output/20251129_024950
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
        "--assets",
        type=str,
        dest="asset_dir",
        help="Path to directory containing image files to use as reference assets for consistency",
    )
    parser.add_argument(
        "--reuse",
        type=str,
        dest="reuse_output_dir",
        help="Path to existing output directory to reuse (skips story/prompt generation, e.g., output/20251129_024950)",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_false",
        dest="save_intermediates",
        help="Don't save intermediate files (story.json, prompts, etc.)",
    )

    args = parser.parse_args()

    run_pipeline(
        user_prompt=args.prompt,
        save_intermediates=args.save_intermediates,
        initial_image_path=args.initial_image_path,
        asset_dir=args.asset_dir,
        reuse_output_dir=args.reuse_output_dir,
    )
