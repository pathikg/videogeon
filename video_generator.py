"""Video generation using Veo 3.1 API and Replicate."""

import time
from pathlib import Path
from typing import Optional

import cv2
import replicate
from google import genai
from google.genai import types
from tqdm import tqdm

import config

# Disable all safety filters
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.OFF,
    ),
]


def load_image_from_file(
    image_path: str,
    client: genai.Client = None,
) -> types.Image:
    """
    Load an image from a file path and convert it to genai Image object.

    Args:
        image_path: Path to the image file
        client: Optional genai client (not used, kept for API compatibility)

    Returns:
        genai Image object (for use with Veo)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Use the built-in from_file method (requires location as keyword argument)
    genai_image = types.Image.from_file(location=str(image_path))

    return genai_image


def generate_initial_image(
    prompt: str,
    client: genai.Client = None,
    save_path: Optional[str] = None,
) -> types.Image:
    """
    Generate initial frame image using Gemini image model.

    Args:
        prompt: Text description for the image
        client: Optional genai client
        save_path: Optional path to save the image

    Returns:
        genai Image object (for use with Veo)
    """
    if client is None:
        client = genai.Client()

    # Create chat for image generation
    chat = client.chats.create(
        model=config.IMAGE_MODEL,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=SAFETY_SETTINGS,
        ),
    )

    response = chat.send_message(prompt)

    # Find the image from response parts
    genai_image = None
    for part in response.parts:
        if image := part.as_image():
            genai_image = image
            # Save to disk if path provided
            if save_path:
                image.save(save_path)
            break

    if genai_image is None:
        raise ValueError("No image generated in response")

    return genai_image


def poll_operation(
    operation, client: genai.Client, desc: str = "Generating"
) -> types.GenerateVideosResponse:
    """
    Poll operation until video generation is complete.

    Args:
        operation: The generation operation to poll
        client: genai client
        desc: Description for progress bar

    Returns:
        Completed operation response
    """
    pbar = tqdm(desc=desc, unit="poll", leave=False)

    while not operation.done:
        time.sleep(config.POLL_INTERVAL)
        operation = client.operations.get(operation)
        pbar.update(1)

    pbar.close()
    return operation


def generate_first_video(
    prompt: str,
    image: types.Image,
    client: genai.Client = None,
    output_path: Optional[str] = None,
    reference_images: Optional[list[types.VideoGenerationReferenceImage]] = None,
) -> types.Video:
    """
    Generate first video segment using image-to-video.

    Args:
        prompt: Video generation prompt
        image: Starting frame image (genai types.Image)
        client: Optional genai client
        output_path: Optional path to save video
        reference_images: Optional list of reference images for asset consistency

    Returns:
        Generated video object
    """
    if client is None:
        client = genai.Client()

    # Create config following the exact pattern from the API example
    config_kwargs = {
        "number_of_videos": 1,
        "duration_seconds": config.VIDEO_DURATION,
    }

    if hasattr(config, "VIDEO_RESOLUTION") and config.VIDEO_RESOLUTION:
        config_kwargs["resolution"] = config.VIDEO_RESOLUTION

    # If we have reference_images, add the initial image to the list and use only reference_images
    # Otherwise, use the image parameter directly
    if reference_images:
        # Add the initial image as a reference image
        initial_reference = types.VideoGenerationReferenceImage(
            image=image, reference_type=types.VideoGenerationReferenceType.ASSET
        )
        all_reference_images = [initial_reference] + reference_images
        config_kwargs["reference_images"] = all_reference_images

        # Use only reference_images, not the image parameter
        operation = client.models.generate_videos(
            model=config.VEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(**config_kwargs),
        )
    else:
        # Use image parameter when no reference images
        operation = client.models.generate_videos(
            model=config.VEO_MODEL,
            prompt=prompt,
            image=image,
            config=types.GenerateVideosConfig(**config_kwargs),
        )

    operation = poll_operation(operation, client, desc="Scene 1")

    # Check if generation succeeded
    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError(f"Video generation failed. Response: {operation.response}")

    video = operation.response.generated_videos[0].video

    if output_path:
        client.files.download(file=video)
        video.save(output_path)

    return video


def extend_video(
    prompt: str,
    previous_video: types.Video,
    client: genai.Client = None,
    output_path: Optional[str] = None,
    scene_number: int = 2,
    reference_images: Optional[list[types.VideoGenerationReferenceImage]] = None,
) -> types.Video:
    """
    Extend a previously generated video with a new scene.

    Args:
        prompt: Video generation prompt for this scene
        previous_video: Video object from previous generation
        client: Optional genai client
        output_path: Optional path to save video
        scene_number: Scene number for progress display
        reference_images: Optional list of reference images for asset consistency

    Returns:
        Generated video object
    """
    if client is None:
        client = genai.Client()

    # Create config following the exact pattern from the API example
    config_kwargs = {
        "number_of_videos": 1,
    }

    if hasattr(config, "VIDEO_RESOLUTION") and config.VIDEO_RESOLUTION:
        config_kwargs["resolution"] = config.VIDEO_RESOLUTION

    # Note: reference_images cannot be used with video parameter (similar to image parameter)
    # Reference images are only supported for the initial video generation
    # if reference_images:
    #     config_kwargs["reference_images"] = reference_images

    operation = client.models.generate_videos(
        model=config.VEO_MODEL,
        prompt=prompt,
        video=previous_video,
        config=types.GenerateVideosConfig(**config_kwargs),
    )

    operation = poll_operation(operation, client, desc=f"Scene {scene_number}")

    # Check if generation succeeded
    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError(
            f"Video extension failed for scene {scene_number}. Response: {operation.response}"
        )

    video = operation.response.generated_videos[0].video

    if output_path:
        client.files.download(file=video)
        video.save(output_path)

    return video


def generate_text_to_video(
    prompt: str, client: genai.Client = None, output_path: Optional[str] = None
) -> types.Video:
    """
    Generate video from text prompt only (no image input).

    Args:
        prompt: Video generation prompt
        client: Optional genai client
        output_path: Optional path to save video

    Returns:
        Generated video object
    """
    if client is None:
        client = genai.Client()

    operation = client.models.generate_videos(
        model=config.VEO_MODEL,
        prompt=prompt,
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            resolution=config.VIDEO_RESOLUTION,
            duration_seconds=config.VIDEO_DURATION,
        ),
    )

    operation = poll_operation(operation, client, desc="Generating video")

    # Check if generation succeeded
    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError(
            f"Text-to-video generation failed. Response: {operation.response}"
        )

    video = operation.response.generated_videos[0].video

    if output_path:
        client.files.download(file=video)
        video.save(output_path)

    return video


def extract_last_frame(video_path: str, output_path: str) -> str:
    """
    Extract the last frame from a video file and save it as an image.

    Args:
        video_path: Path to the input video file
        output_path: Path where the extracted frame will be saved

    Returns:
        Path to the saved frame image
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise RuntimeError(f"Video file has no frames: {video_path}")

    # Seek to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # Read the last frame
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError(f"Failed to read last frame from video: {video_path}")

    # Save the frame
    cv2.imwrite(output_path, frame)
    cap.release()

    return output_path


def generate_video_replicate(
    prompt: str,
    start_image_path: str,
    output_path: str,
    duration: int = None,
    aspect_ratio: str = None,
    negative_prompt: str = "",
) -> str:
    """
    Generate video from image using Replicate API.

    Args:
        prompt: Video generation prompt
        start_image_path: Path to the starting image file
        output_path: Path where the generated video will be saved
        duration: Video duration in seconds (defaults to config.REPLICATE_DURATION)
        aspect_ratio: Aspect ratio string like "16:9" (defaults to config.REPLICATE_ASPECT_RATIO)
        negative_prompt: Negative prompt for video generation

    Returns:
        Path to the generated video file
    """
    if duration is None:
        duration = config.REPLICATE_DURATION
    if aspect_ratio is None:
        aspect_ratio = config.REPLICATE_ASPECT_RATIO

    # Open the image file
    with open(start_image_path, "rb") as image_file:
        # Run the Replicate model
        output = replicate.run(
            config.REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "duration": duration,
                "start_image": image_file,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
            },
        )

        # Download and save the video
        with open(output_path, "wb") as video_file:
            video_file.write(output.read())

    return output_path


def generate_first_video_replicate(
    prompt: str,
    image_path: str,
    output_path: str,
    duration: int = None,
    aspect_ratio: str = None,
    negative_prompt: str = "",
) -> str:
    """
    Generate first video segment using Replicate with initial image.

    Args:
        prompt: Video generation prompt
        image_path: Path to the initial frame image
        output_path: Path where the generated video will be saved
        duration: Video duration in seconds (defaults to config.REPLICATE_DURATION)
        aspect_ratio: Aspect ratio string like "16:9" (defaults to config.REPLICATE_ASPECT_RATIO)
        negative_prompt: Negative prompt for video generation

    Returns:
        Path to the generated video file
    """
    return generate_video_replicate(
        prompt=prompt,
        start_image_path=image_path,
        output_path=output_path,
        duration=duration,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt,
    )
