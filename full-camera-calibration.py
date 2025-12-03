"""
Field Calibration with Rotation Support
Handles cameras that are angled relative to the field direction
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RotatableFieldCalibrator:
    def __init__(self):
        self.field_length = 120  # yards
        self.field_width = 53.3  # yards
    
    def rotate_field_points(self, points, angle_degrees, center=None):
        """
        Rotate field points around a center point
        
        Args:
            points: Nx2 array of field coordinates
            angle_degrees: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center, default is center of points
        """
        if center is None:
            center = np.mean(points, axis=0)
        
        # Convert to radians
        angle_rad = np.radians(angle_degrees)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Translate to origin, rotate, translate back
        points_centered = points - center
        points_rotated = points_centered @ R.T
        points_final = points_rotated + center
        
        return points_final
    
    def compute_homography_with_rotation(self, image_path, tracking_df, play_id, step,
                                        output_dir, rotation_angle=0, corner_params=None):
        """
        Compute homography with field rotation
        
        Args:
            rotation_angle: Degrees to rotate field (positive = counterclockwise)
                           Use this if camera is angled relative to field
            corner_params: Optional manual corner adjustments
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load: {image_path}")
            return None
        
        h, w = img.shape[:2]
        print(f"\nImage size: {w} x {h}")
        print(f"Field rotation: {rotation_angle}°")
        
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
        
        print(f"\nTracking data coverage:")
        print(f"  X (down field): {x_min:.1f} to {x_max:.1f} yards")
        print(f"  Y (sideline):   {y_min:.1f} to {y_max:.1f} yards")
        
        # Field region with margins
        field_x_min = max(0, x_min - 10)
        field_x_max = min(120, x_max + 10)
        field_y_min = 0      # Full width
        field_y_max = 53.3   # Full width
        
        print(f"\nField region:")
        print(f"  X: {field_x_min:.1f} to {field_x_max:.1f} yards")
        print(f"  Y: {field_y_min:.1f} to {field_y_max:.1f} yards (FULL WIDTH)")
        
        # Define field quadrilateral corners (BEFORE rotation)
        field_corners_original = np.array([
            [field_x_min, field_y_min],  # Near sideline, left
            [field_x_max, field_y_min],  # Near sideline, right
            [field_x_max, field_y_max],  # Far sideline, right
            [field_x_min, field_y_max]   # Far sideline, left
        ], dtype=np.float32)
        
        # Apply rotation if needed
        if rotation_angle != 0:
            field_center = np.array([
                (field_x_min + field_x_max) / 2,
                (field_y_min + field_y_max) / 2
            ])
            field_corners = self.rotate_field_points(
                field_corners_original, 
                rotation_angle,
                field_center
            )
            print(f"\n✓ Field corners rotated by {rotation_angle}°")
        else:
            field_corners = field_corners_original
        
        # IMAGE CORNERS
        if corner_params is None:
            # Automatic estimation
            bottom_y = 0.88
            bottom_left_x = 0.05
            bottom_right_x = 0.95
            top_y = 0.12
            perspective_factor = 0.25
            
            top_left_x = bottom_left_x + perspective_factor
            top_right_x = bottom_right_x - perspective_factor
            
            image_corners = np.array([
                [bottom_left_x * w, bottom_y * h],
                [bottom_right_x * w, bottom_y * h],
                [top_right_x * w, top_y * h],
                [top_left_x * w, top_y * h]
            ], dtype=np.float32)
        else:
            # Manual parameters
            image_corners = np.array([
                [corner_params['bottom_left'][0] * w, corner_params['bottom_left'][1] * h],
                [corner_params['bottom_right'][0] * w, corner_params['bottom_right'][1] * h],
                [corner_params['top_right'][0] * w, corner_params['top_right'][1] * h],
                [corner_params['top_left'][0] * w, corner_params['top_left'][1] * h]
            ], dtype=np.float32)
        
        print(f"\nField corners (after rotation):")
        for i, corner in enumerate(field_corners):
            print(f"  Corner {i}: Field ({corner[0]:.1f}, {corner[1]:.1f}) -> "
                  f"Image ({image_corners[i][0]:.0f}, {image_corners[i][1]:.0f})")
        
        # Compute homography
        H, status = cv2.findHomography(field_corners, image_corners, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Failed to compute homography")
            return None
        
        print("\n✓ Homography computed successfully")
        
        # Visualizations
        self.visualize_field_region(
            img, field_corners, field_corners_original, image_corners, H,
            field_x_min, field_x_max, field_y_min, field_y_max,
            rotation_angle, output_dir
        )
        
        return {
            'homography': H,
            'field_corners': field_corners,
            'field_corners_original': field_corners_original,
            'image_corners': image_corners,
            'rotation_angle': rotation_angle,
            'field_bounds': {
                'x_min': field_x_min,
                'x_max': field_x_max,
                'y_min': field_y_min,
                'y_max': field_y_max
            }
        }
    
    def visualize_field_region(self, img, field_corners, field_corners_original,
                               image_corners, H, x_min, x_max, y_min, y_max,
                               rotation_angle, output_dir):
        """Visualize field quadrilateral with rotation"""
        vis_img = img.copy()
        
        # Draw rotated field boundary
        pts = image_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis_img, [pts], True, (0, 255, 255), 3)
        
        # Label corners
        corner_labels = ['Near-Left', 'Near-Right', 'Far-Right', 'Far-Left']
        for i, (corner, label) in enumerate(zip(image_corners, corner_labels)):
            cv2.circle(vis_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(vis_img, label, tuple(corner.astype(int) + np.array([15, 0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw yard lines (accounting for rotation)
        for yard_x in range(int(x_min), int(x_max) + 1, 10):
            # Create yard line in original field coords
            line_field_original = np.array([
                [[yard_x, y_min]],
                [[yard_x, y_max]]
            ], dtype=np.float32)
            
            # Apply rotation if needed
            if rotation_angle != 0:
                field_center = np.array([
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2
                ])
                line_field = self.rotate_field_points(
                    line_field_original.reshape(-1, 2),
                    rotation_angle,
                    field_center
                ).reshape(-1, 1, 2).astype(np.float32)
            else:
                line_field = line_field_original
            
            # Project to image
            line_image = cv2.perspectiveTransform(line_field, H).reshape(-1, 2).astype(np.int32)
            
            # Draw line
            cv2.line(vis_img, tuple(line_image[0]), tuple(line_image[1]), 
                    (255, 255, 0), 2)
            
            # Label
            label_pos = tuple(line_image[0] + np.array([0, 30]))
            cv2.putText(vis_img, f"{yard_x}yd", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw sidelines (accounting for rotation)
        for y_side in [y_min, y_max]:
            sideline_original = np.array([
                [[x_min, y_side]],
                [[x_max, y_side]]
            ], dtype=np.float32)
            
            if rotation_angle != 0:
                field_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                sideline_field = self.rotate_field_points(
                    sideline_original.reshape(-1, 2),
                    rotation_angle,
                    field_center
                ).reshape(-1, 1, 2).astype(np.float32)
            else:
                sideline_field = sideline_original
            
            sideline_image = cv2.perspectiveTransform(sideline_field, H).reshape(-1, 2).astype(np.int32)
            cv2.line(vis_img, tuple(sideline_image[0]), tuple(sideline_image[1]),
                    (0, 255, 255), 3)
        
        # Add info
        cv2.putText(vis_img, f"Field Rotation: {rotation_angle}°",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"Coverage: {x_min:.0f}-{x_max:.0f} yards",
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        output_path = output_dir / f'field_region_rot{rotation_angle}.jpg'
        cv2.imwrite(str(output_path), vis_img)
        print(f"✓ Field visualization saved: {output_path}")
    
    def create_player_overlay(self, image_path, tracking_df, play_id, step,
                             homography_params, output_dir):
        """Create player overlay with rotation"""
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        frame_data = tracking_df[
            (tracking_df['play_id'] == play_id) & 
            (tracking_df['step'] == step)
        ]
        
        # Get field points
        field_points = frame_data[['x_position', 'y_position']].values.astype(np.float32)
        
        # Apply rotation if needed
        if homography_params['rotation_angle'] != 0:
            bounds = homography_params['field_bounds']
            field_center = np.array([
                (bounds['x_min'] + bounds['x_max']) / 2,
                (bounds['y_min'] + bounds['y_max']) / 2
            ])
            field_points = self.rotate_field_points(
                field_points,
                homography_params['rotation_angle'],
                field_center
            )
        
        # Project to image
        image_points = cv2.perspectiveTransform(
            field_points.reshape(-1, 1, 2),
            homography_params['homography']
        ).reshape(-1, 2)
        
        # Draw field boundary (faint)
        corners = homography_params['image_corners'].astype(np.int32)
        cv2.polylines(img, [corners.reshape((-1, 1, 2))], True, (100, 100, 100), 1)
        
        # Draw each player
        for idx, (_, player) in enumerate(frame_data.iterrows()):
            x, y = int(image_points[idx, 0]), int(image_points[idx, 1])
            
            # Skip if outside image
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            # Color by team
            team_str = str(player.get('team', '')).lower()
            if 'home' in team_str:
                color = (255, 100, 100)
            else:
                color = (100, 100, 255)
            
            # Draw marker
            cv2.circle(img, (x, y), 18, color, -1)
            cv2.circle(img, (x, y), 18, (255, 255, 255), 2)
            
            # Jersey number
            jersey = player.get('jersey_number', '?')
            if pd.notna(jersey):
                jersey_text = str(int(jersey))
            else:
                jersey_text = '?'
            
            text_size = cv2.getTextSize(jersey_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            
            cv2.putText(img, jersey_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Info overlay
        rotation = homography_params['rotation_angle']
        cv2.putText(img, f"Rotation: {rotation}° | Play {play_id}, Step {step}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Players: {len(frame_data)}",
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        output_path = output_dir / f'player_overlay_rot{rotation}.jpg'
        cv2.imwrite(str(output_path), img)
        print(f"✓ Player overlay saved: {output_path}")

def test_multiple_rotations(csv_path, image_path, play_id, step, 
                           rotation_angles, output_dir):
    """
    Test multiple rotation angles to find the best fit
    
    Args:
        rotation_angles: List of angles to test (e.g., [0, 5, 10, -5, -10])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    calibrator = RotatableFieldCalibrator()
    
    print("\n" + "="*70)
    print(f"TESTING {len(rotation_angles)} ROTATION ANGLES")
    print("="*70)
    
    results = []
    
    for angle in rotation_angles:
        print(f"\nTesting rotation: {angle}°")
        
        homography_params = calibrator.compute_homography_with_rotation(
            image_path, df, play_id, step, output_dir, rotation_angle=angle
        )
        
        if homography_params:
            calibrator.create_player_overlay(
                image_path, df, play_id, step,
                homography_params, output_dir
            )
            results.append(angle)
    
    print("\n" + "="*70)
    print("ROTATION TEST COMPLETE")
    print("="*70)
    print(f"\nGenerated overlays for rotations: {results}")
    print(f"\nCheck files in {output_dir}/:")
    for angle in results:
        print(f"  - field_region_rot{angle}.jpg")
        print(f"  - player_overlay_rot{angle}.jpg")
    print("\nCompare them to find which rotation angle gives best alignment!")
    print("="*70)

def main():
    print("\n" + "="*70)
    print("FIELD CALIBRATION WITH ROTATION")
    print("="*70)
    
    # Get inputs
    csv_path = "nfl-player-contact-detection/train_player_tracking.csv"
    image_path = "alignment_check_output/first_frame_58221_001269_All29.jpg"
    play_id = 1269
    step = 0
    
    # Rotation mode
    print("\nRotation options:")
    print("1. Single rotation angle")
    print("2. Test multiple angles")
    choice = input("Choice (1 or 2): ").strip()
    
    df = pd.read_csv(csv_path)
    output_dir = Path('./rotated_calibration_output')
    
    if choice == '2':
        # Test multiple angles
        angles_str = input("\nEnter angles to test (e.g., -10,-5,0,5,10): ").strip()
        angles = [int(a.strip()) for a in angles_str.split(',')]
        
        test_multiple_rotations(csv_path, image_path, play_id, step, angles, output_dir)
    else:
        # Single angle
        rotation = float(input("\nRotation angle in degrees (0 = no rotation): ").strip())
        
        calibrator = RotatableFieldCalibrator()
        
        print("\nComputing homography...")
        homography_params = calibrator.compute_homography_with_rotation(
            image_path, df, play_id, step, output_dir, rotation_angle=rotation
        )
        
        if homography_params:
            print("\nCreating player overlay...")
            calibrator.create_player_overlay(
                image_path, df, play_id, step,
                homography_params, output_dir
            )
            
            print("\n" + "="*70)
            print("SUCCESS!")
            print("="*70)
            print(f"\nCheck {output_dir}/ for results")
            print("If alignment is still off, try a different rotation angle")
            print("="*70)

if __name__ == "__main__":
    main()