"""Generate story breakdown from user prompt."""

import json
from pathlib import Path
from google import genai
from google.genai import types

import config

from dotenv import load_dotenv

load_dotenv()

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


def load_prompt(prompt_file: str) -> str:
    """Load prompt template from file."""
    prompt_path = Path("prompts") / prompt_file
    return prompt_path.read_text()


def generate_story(user_prompt: str, client: genai.Client = None) -> dict:
    """
    Generate story breakdown with global context and scene descriptions.

    Args:
        user_prompt: User's story idea (brief or detailed)
        client: Optional genai client (creates one if not provided)

    Returns:
        dict with global_context and scenes[1-4]
    """
    if client is None:
        client = genai.Client()

    system_prompt = load_prompt("01_story_summary.txt")

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "safety_settings": SAFETY_SETTINGS,
        },
    )

    story_data = json.loads(response.text)
    return story_data


def generate_frame_spec(
    story_data: dict, scene_number: int = 1, client: genai.Client = None
) -> dict:
    """
    Generate detailed frame specification for a scene.

    Args:
        story_data: Output from generate_story()
        scene_number: Which scene to generate frame for (1-4)
        client: Optional genai client

    Returns:
        dict with subjects, environment, style, technical, constraints, motion_hint
    """
    if client is None:
        client = genai.Client()

    system_prompt = load_prompt("02_initial_frame.txt")

    # Prepare input for the frame generator
    scene = next(
        (s for s in story_data["scenes"] if s["scene_number"] == scene_number), None
    )
    if not scene:
        raise ValueError(f"Scene {scene_number} not found in story data")

    input_data = {"global_context": story_data["global_context"], "scene": scene}

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=json.dumps(input_data),
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "safety_settings": SAFETY_SETTINGS,
        },
    )

    frame_spec = json.loads(response.text)
    return frame_spec


def generate_video_prompt(
    frame_spec: dict = None, scene_data: dict = None, client: genai.Client = None
) -> dict:
    """
    Convert frame spec / scene data into video model prompt.

    Args:
        frame_spec: Output from generate_frame_spec() (for scene 1)
        scene_data: Scene object from story_data (for scenes 2-4)
        client: Optional genai client

    Returns:
        dict with video_prompt, negative_prompt, parameters
    """
    if client is None:
        client = genai.Client()

    system_prompt = load_prompt("03_video_prompt.txt")

    input_data = {
        "frame_spec": frame_spec,
        "scene_data": scene_data,
        "target_model": "veo",
    }

    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=json.dumps(input_data),
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "safety_settings": SAFETY_SETTINGS,
        },
    )

    video_prompt_data = json.loads(response.text)
    return video_prompt_data
