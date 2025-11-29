# Video Generation Pipeline

Generate videos from story prompts using AI. This project supports two pipelines:
- **pipeline.py**: Uses Google's Gemini API for video generation
- **pipeline_replicate.py**: Uses Replicate for video generation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - For `pipeline.py`: Set `GOOGLE_API_KEY` in a `.env` file
   - For `pipeline_replicate.py`: Set `REPLICATE_API_TOKEN` in a `.env` file

## Usage

### Using pipeline.py (Google Gemini)

Generate a video from a story prompt:
```bash
python pipeline.py "A lonely astronaut discovers a flower on Mars"
```

Use a custom initial image:
```bash
python pipeline.py "Your story prompt" --image path/to/image.png
```

Use reference assets for consistency:
```bash
python pipeline.py "Your story prompt" --assets path/to/assets/folder
```

Reuse an existing output directory (skip story generation):
```bash
python pipeline.py "Story prompt" --reuse output/20251129_024950
```

### Using pipeline_replicate.py (Replicate)

Generate a video from a story prompt:
```bash
python pipeline_replicate.py "A lonely astronaut discovers a flower on Mars"
```

Use a custom initial image:
```bash
python pipeline_replicate.py "Your story prompt" --image path/to/image.png
```

## Output

Both pipelines create a timestamped directory in `output/` containing:
- `story.json` - Generated story breakdown
- `frame_spec.json` - Frame specifications
- `video_prompts.json` - Video generation prompts
- `initial_frame.png` - First frame image
- `scene_1.mp4`, `scene_2.mp4`, etc. - Generated video segments
- Use `concate_videos.py` for replicated generated outputs

## Configuration

Edit `config.py` to adjust:
- Number of scenes
- Video duration
- Model settings
- Output directory

