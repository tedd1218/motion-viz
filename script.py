import cv2
import numpy as np

# ===========================
# 1. Video Setup
# ===========================
input_video = "raw_nfl_clips/mahomes_travis.mp4"  # Replace with your NFL video
output_video = "nfl_motion_highlight.mp4"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
out = cv2.VideoWriter(output_video,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

# ===========================
# 2. Read First Frame
# ===========================
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Optional: smoothing for stability
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

# ===========================
# 3. Process Video Frames
# ===========================
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ----- Motion Detection -----
    # Option A: Simple Frame Difference
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Optional: remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ----- Overlay Motion -----
    motion_frame = frame.copy()
    motion_frame[mask > 0] = [0, 0, 255]  # Highlight motion in red

    # Write frame to output video
    out.write(motion_frame)

    # Update previous frame
    prev_gray = gray.copy()
    frame_count += 1

# ===========================
# 4. Cleanup
# ===========================
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! Processed {frame_count} frames. Output saved as {output_video}")
