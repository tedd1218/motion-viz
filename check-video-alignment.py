"""
Video-Tracking Alignment Checker (Headless Version)
Works on remote servers without display
Saves frames as images instead of showing them
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def check_video_info(video_path):
    """Extract basic information about a video file"""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'path': video_path,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return info

def suggest_plays_for_video(video_path, tracking_df):
    """Suggest plays that might match the video filename"""
    video_name = Path(video_path).stem
    
    # Try to extract play number from filename
    import re
    numbers = re.findall(r'\d+', video_name)
    
    available_plays = sorted(tracking_df['play_id'].unique())
    
    print(f"\nVideo filename: {Path(video_path).name}")
    print(f"Extracted numbers from filename: {numbers}")
    print(f"\nSuggested plays that might match:")
    
    suggestions = []
    for num in numbers:
        try:
            num_int = int(num)
            if num_int in available_plays:
                play_data = tracking_df[tracking_df['play_id'] == num_int]
                steps = play_data['step'].nunique()
                players = play_data['nfl_player_id'].nunique()
                print(f"  - Play {num_int}: {steps} steps, {players} players")
                suggestions.append(num_int)
        except:
            pass
    
    if not suggestions:
        print("  No matching plays found")
        print(f"\nFirst 10 available plays: {available_plays[:10]}")
    
    return suggestions

def list_videos_and_plays(video_folder, tracking_df):
    """List all videos and available plays"""
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4']:
        video_files.extend(glob.glob(str(Path(video_folder) / ext)))
    
    print("\n" + "="*70)
    print("AVAILABLE VIDEOS")
    print("="*70)
    
    for i, vf in enumerate(video_files):
        info = check_video_info(vf)
        print(f"\n[{i}] {Path(vf).name}")
        print(f"    Resolution: {info['width']}x{info['height']}")
        print(f"    FPS: {info['fps']:.2f}")
        print(f"    Frames: {info['frame_count']}")
        print(f"    Duration: {info['duration_sec']}s")
    
    print("\n" + "="*70)
    print("AVAILABLE PLAYS IN TRACKING DATA")
    print("="*70)
    
    plays = sorted(tracking_df['play_id'].unique())
    for i, play in enumerate(plays[:10]):
        play_data = tracking_df[tracking_df['play_id'] == play]
        steps = play_data['step'].nunique()
        players = play_data['nfl_player_id'].nunique()
        print(f"[{i}] Play {play}: {steps} steps, {players} players")
    
    if len(plays) > 10:
        print(f"... and {len(plays)-10} more plays")
    print(f"\nTotal plays available: {len(plays)}")
    
    return video_files

def save_first_frame(video_path, output_path):
    """Save the first frame of a video as an image"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video")
        return False
    
    # Save frame
    cv2.imwrite(str(output_path), frame)
    print(f"\nFirst frame saved to: {output_path}")
    return True

def extract_sample_frames_with_overlay(video_path, tracking_df, play_id, 
                                       output_dir, num_samples=10):
    """
    Extract sample frames with tracking overlay and save as images
    
    Args:
        video_path: Path to video file
        tracking_df: Tracking dataframe
        play_id: Play ID to visualize
        output_dir: Directory to save output images
        num_samples: Number of sample frames to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo: {width}x{height}, {total_frames} frames at {fps:.2f} fps")
    
    # Get play data
    play_data = tracking_df[tracking_df['play_id'] == play_id].sort_values('step')
    
    if len(play_data) == 0:
        print(f"No data found for play {play_id}")
        cap.release()
        return
    
    steps = sorted(play_data['step'].unique())
    print(f"Play has {len(steps)} tracking steps")
    
    # Calculate frame-to-step mapping
    frames_per_step = total_frames / len(steps) if len(steps) > 0 else 1
    print(f"Mapping: ~{frames_per_step:.2f} video frames per tracking step")
    
    # Simple mapping function (adjust based on your camera angle)
    def field_to_image_sideline(x, y):
        """Sideline view - adjust these values based on your camera"""
        img_x = int((x / 120.0) * width)
        img_y = int(height - (y / 53.3) * height)
        return img_x, img_y
    
    def field_to_image_endzone(x, y):
        """Endzone view - adjust these values based on your camera"""
        img_x = int((y / 53.3) * width)
        img_y = int(height - (x / 120.0) * height)
        return img_x, img_y
    
    # Try both mappings
    mapping_functions = {
        'sideline': field_to_image_sideline,
        'endzone': field_to_image_endzone
    }
    
    # Select evenly spaced frames
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    print(f"\nExtracting {num_samples} sample frames...")
    
    for mapping_name, field_to_image in mapping_functions.items():
        print(f"\nGenerating overlays for {mapping_name} view...")
        
        for idx, frame_idx in enumerate(sample_indices):
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_copy = frame.copy()
            
            # Determine current step
            step_idx = int(frame_idx / frames_per_step)
            step_idx = min(step_idx, len(steps) - 1)
            current_step = steps[step_idx]
            
            # Get player positions
            step_data = play_data[play_data['step'] == current_step]
            
            # Draw each player
            for _, player in step_data.iterrows():
                x = player['x_position']
                y = player['y_position']
                
                # Map to image
                img_x, img_y = field_to_image(x, y)
                
                # Skip if outside frame
                if img_x < 0 or img_x >= width or img_y < 0 or img_y >= height:
                    continue
                
                # Color by team
                if 'team' in player and pd.notna(player['team']):
                    team_str = str(player['team']).lower()
                    if 'home' in team_str:
                        color = (255, 0, 0)  # Blue for home
                    elif 'away' in team_str:
                        color = (0, 0, 255)  # Red for away
                    else:
                        color = (0, 255, 0)  # Green for unknown
                else:
                    color = (0, 255, 0)
                
                # Draw marker
                cv2.circle(frame_copy, (img_x, img_y), 15, color, -1)
                cv2.circle(frame_copy, (img_x, img_y), 15, (255, 255, 255), 2)
                
                # Jersey number
                if 'jersey_number' in player and pd.notna(player['jersey_number']):
                    jersey = str(int(player['jersey_number']))
                    cv2.putText(frame_copy, jersey, (img_x-10, img_y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add info overlay
            info_text = [
                f"Mapping: {mapping_name.upper()}",
                f"Video Frame: {frame_idx}/{total_frames}",
                f"Tracking Step: {current_step}",
                f"Players: {len(step_data)}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame_copy, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Save frame
            output_path = output_dir / f"overlay_{mapping_name}_{idx:02d}_frame{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), frame_copy)
            
            if (idx + 1) % 3 == 0:
                print(f"  Saved {idx + 1}/{num_samples} frames")
    
    cap.release()
    
    print(f"\n✓ Done! Saved {num_samples * len(mapping_functions)} images to {output_dir}")
    print("\nReview the images to determine:")
    print("  - Which mapping (sideline vs endzone) looks correct?")
    print("  - Do the dots align with players?")
    print("  - If not, you'll need to adjust the mapping functions")

def create_comparison_grid(output_dir, play_id):
    """Create a side-by-side comparison grid of different mappings"""
    output_dir = Path(output_dir)
    
    # Get all overlay images
    sideline_imgs = sorted(output_dir.glob("overlay_sideline_*.jpg"))
    endzone_imgs = sorted(output_dir.glob("overlay_endzone_*.jpg"))
    
    if len(sideline_imgs) == 0 or len(endzone_imgs) == 0:
        print("No overlay images found")
        return
    
    # Create comparison for first 6 frames
    num_compare = min(6, len(sideline_imgs))
    
    fig, axes = plt.subplots(num_compare, 2, figsize=(20, 5*num_compare))
    
    for i in range(num_compare):
        # Sideline view
        img_s = cv2.imread(str(sideline_imgs[i]))
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_s)
        axes[i, 0].set_title(f"Sideline Mapping - Frame {i}", fontsize=14)
        axes[i, 0].axis('off')
        
        # Endzone view
        img_e = cv2.imread(str(endzone_imgs[i]))
        img_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)
        axes[i, 1].imshow(img_e)
        axes[i, 1].set_title(f"Endzone Mapping - Frame {i}", fontsize=14)
        axes[i, 1].axis('off')
    
    plt.suptitle(f"Tracking Overlay Comparison - Play {play_id}\n"
                 f"LEFT: Sideline Mapping | RIGHT: Endzone Mapping",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "comparison_grid.png"
    plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Comparison grid saved to: {comparison_path}")
    plt.close()

def main():
    """Main function to run alignment check"""
    print("\n" + "="*70)
    print("VIDEO-TRACKING ALIGNMENT CHECKER (Headless)")
    print("="*70)
    
    # Get paths
    csv_path = "nfl-player-contact-detection/train_player_tracking.csv"
    video_folder = "nfl-player-contact-detection/train"
    
    # Load tracking data
    print("\nLoading tracking data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tracking records")
    
    # List videos and plays
    video_files = list_videos_and_plays(video_folder, df)
    
    if len(video_files) == 0:
        print("No videos found!")
        return
    
    # Select video
    video_idx = int(input(f"\nSelect video index (0-{len(video_files)-1}): "))
    selected_video = video_files[video_idx]
    
    # Suggest matching plays
    suggested_plays = suggest_plays_for_video(selected_video, df)
    
    # Create output directory
    output_dir = Path('./alignment_check_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save first frame for camera analysis
    print("\nSaving first frame for camera view analysis...")
    first_frame_path = output_dir / f"first_frame_{Path(selected_video).stem}.jpg"
    save_first_frame(selected_video, first_frame_path)
    
    # Select play
    if suggested_plays:
        use_suggested = input(f"\nUse suggested play {suggested_plays[0]}? (y/n, default y): ").strip().lower()
        if use_suggested != 'n':
            play_id = suggested_plays[0]
            print(f"Using play {play_id}")
        else:
            play_id_input = input("Enter play_id to test: ").strip()
            try:
                play_id = int(play_id_input)
            except ValueError:
                play_id = play_id_input
    else:
        play_id_input = input("\nEnter play_id to test: ").strip()
        try:
            play_id = int(play_id_input)
        except ValueError:
            play_id = play_id_input
    
    # Check if play exists
    if play_id not in df['play_id'].values:
        print(f"Play {play_id} not found in tracking data!")
        print(f"\nAvailable plays: {sorted(df['play_id'].unique())[:20]}...")
        return
    
    # Number of sample frames
    num_samples = int(input("\nNumber of sample frames to extract (default 10): ").strip() or "10")
    
    # Extract frames with overlay
    print("\nExtracting sample frames with tracking overlay...")
    extract_sample_frames_with_overlay(
        selected_video, 
        df, 
        play_id, 
        output_dir,
        num_samples=num_samples
    )
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    create_comparison_grid(output_dir, play_id)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    print(f"All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - first_frame_*.jpg - First frame of video for camera analysis")
    print(f"  - overlay_sideline_*.jpg - Frames with sideline mapping")
    print(f"  - overlay_endzone_*.jpg - Frames with endzone mapping")
    print(f"  - comparison_grid.png - Side-by-side comparison")
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the comparison_grid.png")
    print("2. Determine which mapping (sideline or endzone) is more accurate")
    print("3. Check if dots align with players")
    print("4. If misaligned, adjust the field_to_image functions in the code")
    print("5. Once aligned, proceed to camera calibration")
    print("="*70)

if __name__ == "__main__":
    main()