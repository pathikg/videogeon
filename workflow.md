User prompt --> Video 

## Constraints
- Video segment duration: 5 or 10 seconds (Replicate model limitation)
- Total scenes: 4
- Duration values are automatically clamped to valid Replicate values (5 or 10)
- Aspect ratios are automatically clamped to valid values ("16:9", "9:16", "1:1")

## Pipeline

```pseudocode
FUNCTION run_pipeline(user_prompt):
    // Step 1: Generate story breakdown
    story_data = generate_story(user_prompt)
    // Output: {global_context, scenes[1-4]}
    
    // Step 2: Generate frame specification for Scene 1
    frame_spec = generate_frame_spec(story_data, scene_number=1)
    // Output: Detailed frame specification for initial scene
    
    // Step 3: Generate video prompts for all scenes
    video_prompts = []
    FOR scene_num IN [1, 2, 3, 4]:
        IF scene_num == 1:
            prompt_data = generate_video_prompt(frame_spec, scene_data)
        ELSE:
            prompt_data = generate_video_prompt(scene_data=scenes[scene_num])
        video_prompts.append(prompt_data)
    
    // Step 4: Generate initial image for Scene 1
    initial_image = generate_initial_image(video_prompts[0].video_prompt)
    SAVE initial_image TO initial_frame.png
    
    // Step 5: Generate videos iteratively
    current_image = initial_image
    
    // Scene 1: Image-to-Video
    video_1 = generate_first_video_replicate(
        prompt=video_prompts[0].video_prompt,
        image_path=initial_image,
        duration=clamp_duration(video_prompts[0].duration),
        aspect_ratio=clamp_aspect_ratio(video_prompts[0].aspect_ratio)
    )
    SAVE video_1 TO scene_1.mp4
    
    // Scenes 2-4: Extract last frame and generate next video
    FOR scene_num IN [2, 3, 4]:
        // Extract last frame from previous video
        last_frame = extract_last_frame(video_{scene_num-1}.mp4)
        SAVE last_frame TO scene_{scene_num-1}_last_frame.png
        
        // Generate next video using extracted frame
        video_{scene_num} = generate_video_replicate(
            prompt=video_prompts[scene_num-1].video_prompt,
            start_image_path=last_frame,
            duration=clamp_duration(video_prompts[scene_num-1].duration),
            aspect_ratio=clamp_aspect_ratio(video_prompts[scene_num-1].aspect_ratio)
        )
        SAVE video_{scene_num} TO scene_{scene_num}.mp4
    
    // Step 6: Concatenate all videos
    final_video = concatenate_videos([scene_1.mp4, scene_2.mp4, scene_3.mp4, scene_4.mp4])
    SAVE final_video TO final_video.mp4
    
    RETURN output_directory
END FUNCTION

FUNCTION clamp_duration(duration):
    IF duration <= 7:
        RETURN 5
    ELSE:
        RETURN 10
END FUNCTION

FUNCTION clamp_aspect_ratio(aspect_ratio):
    valid_ratios = ["16:9", "9:16", "1:1"]
    IF aspect_ratio IN valid_ratios:
        RETURN aspect_ratio
    
    // Parse and map to nearest valid ratio
    ratio = parse_aspect_ratio(aspect_ratio)
    IF ratio > 1.5:
        RETURN "16:9"
    ELSE IF ratio < 0.6:
        RETURN "9:16"
    ELSE:
        RETURN "1:1"
END FUNCTION

FUNCTION extract_last_frame(video_path):
    video = OPEN video_path
    total_frames = GET_FRAME_COUNT(video)
    last_frame = READ_FRAME(video, total_frames - 1)
    RETURN last_frame
END FUNCTION
```

## Scene 1 (Text-to-Video)
- Uses 02_initial_frame → 03_video_prompt → T2V generation

## Scenes 2-4 (Image-to-Video)  
- Uses scene[N].description + last_frame → 03_video_prompt → I2V generation
- Last frame provides visual continuity

## Prompts Directory
- `prompts/01_story_summary.txt` - Global story + 4 scene descriptions
- `prompts/02_initial_frame.txt` - Scene 1 → first frame specification  
- `prompts/03_video_prompt.txt` - Frame spec → video model prompt

## Code Structure
```
config.py              # Settings (model, duration, resolution, Replicate config)
story_generator.py      # LLM calls for story/frame/prompt generation (Gemini)
video_generator.py     # Video generation (Veo 3.1 API + Replicate functions)
pipeline.py            # Main orchestrator (Veo pipeline)
pipeline_replicate.py   # Replicate-based pipeline orchestrator
concat_videos.py       # Video concatenation utility
```

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export GEMINI_API_KEY="your-key"
export REPLICATE_API_TOKEN="your-token"

# Run Veo pipeline
python pipeline.py "A lonely astronaut discovers a flower on Mars"

# Run Replicate pipeline
python pipeline_replicate.py "A lonely astronaut discovers a flower on Mars"

# Concatenate generated videos (for replicate)
python concat_videos.py [output_directory]
```