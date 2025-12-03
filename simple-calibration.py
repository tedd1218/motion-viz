"""
Simple Homography-Based Field Alignment
Maps field coordinates directly to image using 2D homography
This is simpler and more robust than full 3D calibration
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def estimate_homography_from_tracking(image_path, tracking_df, play_id, step, 
                                      output_dir, manual_corners=None):
    """
    Estimate homography (field to image) using tracking data
    
    Strategy:
    1. Find extreme player positions in tracking data
    2. Manually identify 4 corners in the image that match a field quadrilateral
    3. Compute homography matrix
    4. Use to project all tracking points
    
    Args:
        image_path: Path to video frame
        tracking_df: Tracking dataframe
        play_id: Play ID
        step: Frame step
        output_dir: Output directory
        manual_corners: If provided, dict with 'field' and 'image' corner coordinates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Get tracking data
    frame_data = tracking_df[
        (tracking_df['play_id'] == play_id) & 
        (tracking_df['step'] == step)
    ]
    
    if len(frame_data) == 0:
        print(f"No tracking data found")
        return None
    
    # Analyze field coverage
    x_min, x_max = frame_data['x_position'].min(), frame_data['x_position'].max()
    y_min, y_max = frame_data['y_position'].min(), frame_data['y_position'].max()
    
    print(f"\nField coverage in tracking:")
    print(f"  X (length): {x_min:.1f} to {x_max:.1f} yards")
    print(f"  Y (width):  {y_min:.1f} to {y_max:.1f} yards")
    
    if manual_corners is None:
        # Use automatic estimation based on field geometry
        # Expand the tracked region slightly to get full visible field
        x_margin = (x_max - x_min) * 0.2
        y_margin = (y_max - y_min) * 0.1
        
        field_x_min = max(0, x_min - x_margin)
        field_x_max = min(120, x_max + x_margin)
        field_y_min = max(0, y_min - y_margin)
        field_y_max = min(53.3, y_max + y_margin)
        
        # Define field quadrilateral (4 corners)
        field_corners = np.array([
            [field_x_min, field_y_min],  # Bottom-left (near sideline, left)
            [field_x_max, field_y_min],  # Bottom-right (near sideline, right)
            [field_x_max, field_y_max],  # Top-right (far sideline, right)
            [field_x_min, field_y_max]   # Top-left (far sideline, left)
        ], dtype=np.float32)
        
        # Estimate image corners based on typical sideline camera view
        # Near sideline: bottom of image
        # Far sideline: top of image (smaller due to perspective)
        
        # Adjust these ratios to match your camera view!
        bottom_margin = 0.15  # How much of bottom is not field
        top_margin = 0.05     # How much of top is not field
        left_margin = 0.1     # How much of left side
        right_margin = 0.1    # How much of right side
        
        # Perspective factor: far side appears smaller
        perspective_shrink = 0.2
        
        image_corners = np.array([
            [w * left_margin, h * (1 - bottom_margin)],  # Bottom-left
            [w * (1 - right_margin), h * (1 - bottom_margin)],  # Bottom-right
            [w * (1 - right_margin - perspective_shrink), h * top_margin],  # Top-right
            [w * (left_margin + perspective_shrink), h * top_margin]  # Top-left
        ], dtype=np.float32)
    else:
        field_corners = np.array(manual_corners['field'], dtype=np.float32)
        image_corners = np.array(manual_corners['image'], dtype=np.float32)
    
    # Compute homography: field -> image
    H, status = cv2.findHomography(field_corners, image_corners, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Failed to compute homography")
        return None
    
    print("\n✓ Homography computed successfully")
    
    # Visualize the field corners used
    vis_img = img.copy()
    for i, corner in enumerate(image_corners):
        cv2.circle(vis_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(vis_img, f"C{i}", tuple(corner.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw quadrilateral
    pts = image_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 3)
    
    cv2.imwrite(str(output_dir / 'field_quadrilateral.jpg'), vis_img)
    print(f"Field quadrilateral saved to: field_quadrilateral.jpg")
    
    return {
        'homography': H,
        'field_corners': field_corners,
        'image_corners': image_corners,
        'field_bounds': {
            'x_min': field_corners[:, 0].min(),
            'x_max': field_corners[:, 0].max(),
            'y_min': field_corners[:, 1].min(),
            'y_max': field_corners[:, 1].max()
        }
    }

def project_tracking_with_homography(tracking_df, step, homography_params):
    """
    Project tracking positions using homography
    """
    frame_data = tracking_df[tracking_df['step'] == step]
    
    # Get all field positions
    field_points = frame_data[['x_position', 'y_position']].values.astype(np.float32)
    
    # Apply homography
    image_points = cv2.perspectiveTransform(
        field_points.reshape(-1, 1, 2),
        homography_params['homography']
    ).reshape(-1, 2)
    
    # Create result list
    projected = []
    for idx, (_, player) in enumerate(frame_data.iterrows()):
        projected.append({
            'player_id': player['nfl_player_id'],
            'img_x': int(image_points[idx, 0]),
            'img_y': int(image_points[idx, 1]),
            'jersey': player.get('jersey_number', '?'),
            'team': player.get('team', 'unknown'),
            'field_x': player['x_position'],
            'field_y': player['y_position']
        })
    
    return projected

def create_overlay_with_homography(image_path, tracking_df, play_id, step,
                                   homography_params, output_path):
    """
    Create overlay using homography projection
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    # Project players
    projected = project_tracking_with_homography(
        tracking_df[tracking_df['play_id'] == play_id],
        step,
        homography_params
    )
    
    # Draw field boundary
    corners = homography_params['image_corners'].astype(np.int32)
    cv2.polylines(img, [corners.reshape((-1, 1, 2))], True, (255, 255, 0), 2)
    
    # Draw yard lines
    field_bounds = homography_params['field_bounds']
    for yard_x in range(int(field_bounds['x_min']), int(field_bounds['x_max']) + 1, 10):
        # Project yard line
        line_field = np.array([
            [[yard_x, field_bounds['y_min']]],
            [[yard_x, field_bounds['y_max']]]
        ], dtype=np.float32)
        
        line_image = cv2.perspectiveTransform(
            line_field,
            homography_params['homography']
        ).reshape(-1, 2).astype(np.int32)
        
        cv2.line(img, tuple(line_image[0]), tuple(line_image[1]), (255, 255, 0), 1)
        
        # Label
        cv2.putText(img, str(yard_x), tuple(line_image[0]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw players
    for player in projected:
        x, y = player['img_x'], player['img_y']
        
        # Skip if outside image
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            continue
        
        # Color by team
        team_str = str(player['team']).lower()
        if 'home' in team_str:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red
        
        # Draw
        cv2.circle(img, (x, y), 15, color, -1)
        cv2.circle(img, (x, y), 15, (255, 255, 255), 2)
        
        # Jersey
        jersey = str(int(player['jersey'])) if player['jersey'] != '?' else '?'
        cv2.putText(img, jersey, (x-10, y+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Info overlay
    cv2.putText(img, f"Homography Projection - Play {play_id}, Step {step}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Players: {len(projected)}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), img)
    print(f"✓ Overlay saved to: {output_path}")

def batch_process_frames(image_folder, tracking_df, play_id, homography_params, 
                         output_dir, num_frames=10):
    """
    Apply homography to multiple frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    play_data = tracking_df[tracking_df['play_id'] == play_id]
    steps = sorted(play_data['step'].unique())
    
    # Sample evenly
    step_indices = np.linspace(0, len(steps)-1, num_frames, dtype=int)
    selected_steps = [steps[i] for i in step_indices]
    
    print(f"\nProcessing {num_frames} frames...")
    
    # Assume images are named similarly
    # You'll need to adjust this based on your naming convention
    for idx, step in enumerate(selected_steps):
        # Find corresponding image (this is a guess, adjust as needed)
        image_files = list(Path(image_folder).glob(f"*{play_id}*.jpg"))
        
        if len(image_files) == 0:
            print(f"No images found for play {play_id}")
            break
        
        # Use first image as template (adjust frame selection logic as needed)
        image_path = image_files[0]
        
        output_path = output_dir / f"homography_overlay_step{step:04d}.jpg"
        
        create_overlay_with_homography(
            image_path, tracking_df, play_id, step,
            homography_params, output_path
        )
        
        if (idx + 1) % 3 == 0:
            print(f"  Processed {idx + 1}/{num_frames}")
    
    print(f"\n✓ All frames saved to {output_dir}")

def interactive_corner_adjustment():
    """
    Help user interactively adjust corner positions
    """
    print("\n" + "="*70)
    print("INTERACTIVE CORNER ADJUSTMENT")
    print("="*70)
    print("\nThe automatic homography uses estimated corner positions.")
    print("If alignment is poor, you can manually adjust the corners.")
    print("\nDefault corner positions (as fractions of image width/height):")
    print("  Bottom-left:  (0.1 * w, 0.85 * h)")
    print("  Bottom-right: (0.9 * w, 0.85 * h)")
    print("  Top-right:    (0.7 * w, 0.05 * h)")
    print("  Top-left:     (0.3 * w, 0.05 * h)")
    print("\nAdjust these values based on your camera view!")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("HOMOGRAPHY-BASED FIELD ALIGNMENT")
    print("="*70)
    
    # Get inputs
    csv_path = "nfl-player-contact-detection/train_player_tracking.csv"
    
    # Check if using existing extracted frame or video
    use_existing = input("Use existing extracted frame? (y/n): ").strip().lower()
    
    if use_existing == 'y':
        image_path = input("Path to frame image: ").strip()
    else:
        print("Please extract a frame first using check_alignment_headless.py")
        return
    
    play_id = int(input("Play ID: ").strip())
    step = int(input("Step number: ").strip())
    
    # Load data
    df = pd.read_csv(csv_path)
    
    output_dir = Path('./homography_output')
    
    # Estimate homography
    print("\nEstimating homography...")
    homography_params = estimate_homography_from_tracking(
        image_path, df, play_id, step, output_dir
    )
    
    if homography_params:
        # Create overlay
        print("\nCreating overlay...")
        create_overlay_with_homography(
            image_path, df, play_id, step, homography_params,
            output_dir / 'homography_overlay.jpg'
        )
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nCheck these files in {output_dir}:")
        print("  - field_quadrilateral.jpg (shows estimated field region)")
        print("  - homography_overlay.jpg (players projected with homography)")
        print("\nIf alignment still looks off:")
        print("  1. Adjust corner estimation parameters in the code")
        print("  2. Or provide manual corner coordinates")
        print("="*70)
    else:
        print("\nHomography estimation failed")

if __name__ == "__main__":
    main()