# MediaPipe Video Filter Specs

## Filter Specs

1. `background_blur`: Segment person with MediaPipe, blur background only, keep person sharp.
2. `virtual_glasses`: Track face landmarks, place glasses asset over both eyes, scale with face width.
3. `neon_face_mesh`: Draw glowing face mesh lines using Face Landmarker points.
4. `beauty_soften`: Smooth skin area inside face oval, preserve eyes, lips, eyebrows.
5. `cartoon_face`: Posterize colors, add black edge outlines around face and body.
6. `hand_magic_trail`: Track index fingertips, draw fading particle trail while hand moves.
7. `pose_aura`: Track body pose, draw animated glow around torso, arms, and head.
8. `face_sticker`: Attach sticker to forehead or cheeks, follow head rotation and scale.
9. `background_replace`: Replace background with image/video while preserving person mask.
10. `background_blur_lite`: Optimized low-cost blur mode, target FPS > 24 on webcam input.

## Technical Requirements

11. Use Python 3.10+.
12. Use OpenCV for webcam capture, frame display, keyboard input, and video writing.
13. Use MediaPipe Tasks API for face, hand, pose, and segmentation detection.
14. App must support webcam index config, default `0`.
15. App must support resolution config: `640x480`, `1280x720`.
16. Pipeline must convert OpenCV `BGR` frames to MediaPipe `SRGB`.
17. Pipeline must use monotonic timestamps for MediaPipe video mode.
18. Filter system must expose common interface: `process(frame, timestamp_ms) -> frame`.
19. User must switch filters at runtime with keyboard keys `1-0`.
20. App must show live FPS counter on screen.
21. Optimized mode must maintain average FPS > 24 for 60 seconds at `640x480`.
22. Heavy filters must allow frame skipping or lower-resolution inference.
23. Assets must load from local `assets/` folder: stickers, backgrounds, overlays.
24. App must fail gracefully when webcam, model, or asset file missing.
25. Code must keep filter modules separate, one file per filter, no file over 200 LOC unless justified.

## Runtime Mapping

- Command: `uv run python main.py filters`.
- Default optimized filter: key `0`, `background_blur_lite`, `640x480`, `frame_skip=1`, `inference_scale=0.5`.
- Switch filters with keys `1` through `0`; press `q` to quit.
- Use `--record-output output.mp4` for OpenCV video writing.
- Use `--benchmark-seconds 60 --video-output none` to measure average FPS for requirement 21 on local webcam hardware.
