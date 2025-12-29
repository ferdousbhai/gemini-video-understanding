#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0.0",
# ]
# ///
"""
Gemini Video Understanding - Analyze videos using Google's Gemini API.

Supports:
- Local video files (any size via File API)
- YouTube URLs
- Inline video data for small files (<20MB)
- Custom frame rate sampling
- Video clipping with start/end timestamps
"""

import argparse
import os
import sys
import time
from pathlib import Path


def get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(file_path).suffix.lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/mov",
        ".avi": "video/avi",
        ".flv": "video/x-flv",
        ".mpg4": "video/mp4",
        ".webm": "video/webm",
        ".wmv": "video/wmv",
        ".3gp": "video/3gpp",
        ".3gpp": "video/3gpp",
        ".mkv": "video/x-matroska",
    }
    return mime_types.get(ext, "video/mp4")


def is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    youtube_patterns = [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/shorts/",
        "youtube.com/embed/",
    ]
    return any(pattern in source for pattern in youtube_patterns)


def wait_for_file_processing(client, file) -> None:
    """Wait for uploaded file to finish processing."""
    while file.state.name == "PROCESSING":
        print("Processing video...", file=sys.stderr)
        time.sleep(5)
        file = client.files.get(name=file.name)

    if file.state.name == "FAILED":
        raise RuntimeError(f"File processing failed: {file.state.name}")


def understand_video(
    source: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    fps: float | None = None,
    start_offset: float | None = None,
    end_offset: float | None = None,
    api_key: str | None = None,
) -> str:
    """
    Analyze a video using Gemini API.

    Args:
        source: Path to local video file or YouTube URL
        prompt: Question or instruction for analyzing the video
        model: Gemini model to use (default: gemini-3-pro-preview)
        fps: Custom frame rate sampling (default: 1 fps)
        start_offset: Start time in seconds for video clipping
        end_offset: End time in seconds for video clipping
        api_key: Optional API key (falls back to GEMINI_API_KEY env var)

    Returns:
        The model's response text
    """
    from google import genai
    from google.genai import types

    # Resolve API key
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_api_key:
        print("Error: No API key provided.", file=sys.stderr)
        print(
            "Set GEMINI_API_KEY environment variable or use --api-key argument.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=resolved_api_key)

    # Build content parts
    parts = []

    # Build video metadata if custom options specified
    video_metadata = None
    if fps is not None or start_offset is not None or end_offset is not None:
        metadata_kwargs = {}
        if fps is not None:
            metadata_kwargs["fps"] = fps
        if start_offset is not None:
            metadata_kwargs["start_offset"] = f"{start_offset}s"
        if end_offset is not None:
            metadata_kwargs["end_offset"] = f"{end_offset}s"
        video_metadata = types.VideoMetadata(**metadata_kwargs)

    if is_youtube_url(source):
        # YouTube URL - use file_data with URI
        file_data_kwargs = {"file_uri": source}
        if video_metadata:
            file_data_kwargs["video_metadata"] = video_metadata
        parts.append(types.Part(file_data=types.FileData(**file_data_kwargs)))
    else:
        # Local file - upload via File API
        video_path = Path(source)
        if not video_path.exists():
            print(f"Error: Video file not found: {source}", file=sys.stderr)
            sys.exit(1)

        print(f"Uploading video: {video_path.name}...", file=sys.stderr)

        # Upload file
        uploaded_file = client.files.upload(file=str(video_path))

        # Wait for processing
        wait_for_file_processing(client, uploaded_file)
        print("Video processed successfully.", file=sys.stderr)

        # Create file reference with optional metadata
        if video_metadata:
            parts.append(
                types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded_file.uri, video_metadata=video_metadata
                    )
                )
            )
        else:
            parts.append(uploaded_file)

    # Add the text prompt
    parts.append(types.Part(text=prompt))

    # Generate response
    try:
        response = client.models.generate_content(
            model=model,
            contents=types.Content(parts=parts),
        )
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract text response
    if response.candidates and response.candidates[0].content.parts:
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        if text_parts:
            return "\n".join(text_parts)

    print("Error: No text response from the model.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze videos using Google's Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a local video
  %(prog)s --source video.mp4 --prompt "Describe what happens in this video"

  # Analyze a YouTube video
  %(prog)s --source "https://youtube.com/watch?v=..." --prompt "Summarize this video"

  # Extract specific information with timestamps
  %(prog)s --source video.mp4 --prompt "List all text shown on screen with timestamps"

  # Analyze a portion of a video (30s to 60s)
  %(prog)s --source video.mp4 --prompt "What happens?" --start 30 --end 60

  # Use custom frame rate (2 fps for more detail)
  %(prog)s --source video.mp4 --prompt "Describe the action" --fps 2

        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to video file or YouTube URL",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Question or instruction for analyzing the video",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-pro-preview",
        help="Gemini model to use (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Custom frame rate sampling (default: 1 fps)",
    )
    parser.add_argument(
        "--start",
        type=float,
        dest="start_offset",
        help="Start time in seconds for video clipping",
    )
    parser.add_argument(
        "--end",
        type=float,
        dest="end_offset",
        help="End time in seconds for video clipping",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )

    args = parser.parse_args()

    result = understand_video(
        source=args.source,
        prompt=args.prompt,
        model=args.model,
        fps=args.fps,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        api_key=args.api_key,
    )

    print(result)


if __name__ == "__main__":
    main()
