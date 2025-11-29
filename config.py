"""Configuration settings for video generation pipeline."""

# Model settings
VEO_MODEL = "veo-3.1-generate-preview"
IMAGE_MODEL = "gemini-3-pro-image-preview"
LLM_MODEL = "gemini-2.5-flash"

# Video settings
VIDEO_DURATION = 8  # seconds
VIDEO_RESOLUTION = "720p"
NUM_SCENES = 4
TOTAL_DURATION = VIDEO_DURATION * NUM_SCENES  # 32 seconds

# Generation settings
GENERATE_AUDIO = False
POLL_INTERVAL = 10  # seconds

# Output settings
OUTPUT_DIR = "output"

# Replicate settings
REPLICATE_MODEL = "kwaivgi/kling-v2.5-turbo-pro"
REPLICATE_DURATION = 5  # seconds
REPLICATE_ASPECT_RATIO = "16:9"
