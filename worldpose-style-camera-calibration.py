"""
Full 3D Camera Calibration - WorldPose Style
Uses field markings to calibrate camera intrinsics and extrinsics
Then projects field lines and players accurately in 3D
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json

class WorldPose3DCalibrator:
    """
    Full 3D camera calibration using NFL field markings
    Following WorldPose methodology
    """
    
    def __init__(self):
        self.field_length = 120  # yards
        self.field_width = 53.3  # yards
        
    def get_field_calibration_points(self, visible_yards_range=None):
        """
        Get known 3D points on the field for calibration
        
        Args:
            visible_yards_range: (x_min, x_max) visible yard range
                                e.g., (20, 50) for yards 20-50
        
        Returns:
            List of calibration points with names and 3D positions
        """
        if visible_yards_range is None:
            x_min, x_max = 10, 110
        else:
            x_min, x_max = visible_yards_range
        
        points = []
        
        # Key intersection points that are easy to identify in images
        
        # 1. Yard line intersections with near sideline
        for x in range(int(x_min), int(x_max) + 1, 10):
            points.append({
                'name': f'{x}yd_near_sideline',
                'coords_3d': [x, 0, 0],  # Near sideline
                'type': 'yard_sideline'
            })
        
        # 2. Yard line intersections with far sideline
        for x in range(int(x_min), int(x_max) + 1, 10):
            points.append({
                'name': f'{x}yd_far_sideline',
                'coords_3d': [x, 53.3, 0],  # Far sideline
                'type': 'yard_sideline'
            })
        
        # 3. Yard line intersections with left hash mark
        for x in range(int(x_min), int(x_max) + 1, 10):
            points.append({
                'name': f'{x}yd_left_hash',
                'coords_3d': [x, 23.36, 0],
                'type': 'yard_hash'
            })
        
        # 4. Yard line intersections with right hash mark
        for x in range(int(x_min), int(x_max) + 1, 10):
            points.append({
                'name': f'{x}yd_right_hash',
                'coords_3d': [x, 29.94, 0],
                'type': 'yard_hash'
            })
        
        # 5. Goal line corners (very visible)
        for x in [10, 110]:
            points.append({
                'name': f'goal_{x}_near',
                'coords_3d': [x, 0, 0],
                'type': 'goal_corner'
            })
            points.append({
                'name': f'goal_{x}_far',
                'coords_3d': [x, 53.3, 0],
                'type': 'goal_corner'
            })
        
        return points
    
    def calibrate_from_correspondences(self, points_3d, points_2d, image_size):
        """
        Calibrate camera using 3D-2D point correspondences
        
        Args:
            points_3d: Nx3 array of 3D world coordinates (yards)
            points_2d: Nx2 array of 2D image coordinates (pixels)
            image_size: (width, height) of image
        
        Returns:
            camera_params: Dict with calibration results
        """
        w, h = image_size
        
        # Convert to numpy arrays
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        print(f"\nCalibrating with {len(points_3d)} point correspondences...")
        
        # Initial camera matrix guess
        # Focal length typically 1.0-1.5x image width for broadcast cameras
        focal_length = w * 1.2
        
        camera_matrix_init = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Solve PnP to get camera pose (rotation and translation)
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            camera_matrix_init,
            None,  # No distortion initially
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("❌ Initial PnP failed")
            return None
        
        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            points_3d,
            points_2d,
            camera_matrix_init,
            None,
            rvec,
            tvec
        )
        
        # Calculate reprojection error
        projected_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, camera_matrix_init, None
        )
        projected_2d = projected_2d.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_2d - points_2d, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\n✓ Calibration successful!")
        print(f"  Mean reprojection error: {mean_error:.2f} pixels")
        print(f"  Max reprojection error: {max_error:.2f} pixels")
        
        # Extract rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Calculate camera position in world coordinates
        camera_position = -rotation_matrix.T @ tvec
        
        print(f"\n  Camera position (world coords):")
        print(f"    X: {camera_position[0][0]:.1f} yards")
        print(f"    Y: {camera_position[1][0]:.1f} yards")
        print(f"    Z: {camera_position[2][0]:.1f} yards (height above field)")
        
        # Calculate viewing angles
        # Camera looks along -Z axis in camera frame
        camera_forward = rotation_matrix[:, 2]
        
        # Tilt angle (angle from horizontal)
        tilt_angle = np.degrees(np.arcsin(-camera_forward[2]))
        
        # Pan angle (rotation around vertical axis)
        pan_angle = np.degrees(np.arctan2(camera_forward[1], camera_forward[0]))
        
        print(f"\n  Camera orientation:")
        print(f"    Tilt: {tilt_angle:.1f}° (from horizontal)")
        print(f"    Pan: {pan_angle:.1f}° (from +X axis)")
        
        return {
            'camera_matrix': camera_matrix_init,
            'dist_coeffs': None,
            'rvec': rvec,
            'tvec': tvec,
            'rotation_matrix': rotation_matrix,
            'camera_position': camera_position,
            'tilt_angle': tilt_angle,
            'pan_angle': pan_angle,
            'mean_error': mean_error,
            'max_error': max_error,
            'errors': errors,
            'success': True
        }
    
    def save_calibration(self, camera_params, output_path):
        """Save calibration to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        save_dict = {
            'camera_matrix': camera_params['camera_matrix'].tolist(),
            'rvec': camera_params['rvec'].tolist(),
            'tvec': camera_params['tvec'].tolist(),
            'rotation_matrix': camera_params['rotation_matrix'].tolist(),
            'camera_position': camera_params['camera_position'].tolist(),
            'tilt_angle': float(camera_params['tilt_angle']),
            'pan_angle': float(camera_params['pan_angle']),
            'mean_error': float(camera_params['mean_error']),
            'max_error': float(camera_params['max_error'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"\n✓ Calibration saved to {output_path}")
    
    def load_calibration(self, json_path):
        """Load calibration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return {
            'camera_matrix': np.array(data['camera_matrix'], dtype=np.float32),
            'rvec': np.array(data['rvec'], dtype=np.float32),
            'tvec': np.array(data['tvec'], dtype=np.float32),
            'rotation_matrix': np.array(data['rotation_matrix'], dtype=np.float32),
            'camera_position': np.array(data['camera_position'], dtype=np.float32),
            'tilt_angle': data['tilt_angle'],
            'pan_angle': data['pan_angle'],
            'mean_error': data['mean_error'],
            'max_error': data['max_error'],
            'dist_coeffs': None,
            'success': True
        }


class FieldLineRenderer:
    """Render field lines using 3D projection"""
    
    def __init__(self):
        self.field_length = 120
        self.field_width = 53.3
    
    def get_field_lines_3d(self):
        """Get all field line segments in 3D"""
        lines = []
        
        # Sidelines
        lines.append({
            'name': 'near_sideline',
            'start': [0, 0, 0],
            'end': [120, 0, 0],
            'color': (255, 255, 255),
            'thickness': 2
        })
        lines.append({
            'name': 'far_sideline',
            'start': [0, 53.3, 0],
            'end': [120, 53.3, 0],
            'color': (255, 255, 255),
            'thickness': 2
        })
        
        # Yard lines (every 5 yards)
        for x in range(5, 120, 5):
            thickness = 3 if x in [10, 110] else 1  # Goal lines thicker
            lines.append({
                'name': f'yard_{x}',
                'start': [x, 0, 0],
                'end': [x, 53.3, 0],
                'color': (255, 255, 255),
                'thickness': thickness
            })
        
        # Hash marks (every 1 yard)
        for x in range(10, 111, 1):
            # Left hash
            lines.append({
                'name': f'hash_L_{x}',
                'start': [x, 23.1, 0],
                'end': [x, 23.6, 0],
                'color': (255, 255, 255),
                'thickness': 1
            })
            # Right hash
            lines.append({
                'name': f'hash_R_{x}',
                'start': [x, 29.7, 0],
                'end': [x, 30.2, 0],
                'color': (255, 255, 255),
                'thickness': 1
            })
        
        return lines
    
    def draw_field_lines(self, image, camera_params, alpha=0.3, 
                        line_filter=None):
        """
        Draw field lines on image using 3D projection
        
        Args:
            image: Input image
            camera_params: Camera calibration parameters
            alpha: Transparency (0=invisible, 1=opaque)
            line_filter: Optional function to filter lines
        """
        overlay = image.copy()
        lines = self.get_field_lines_3d()
        
        if line_filter:
            lines = [l for l in lines if line_filter(l)]
        
        h, w = image.shape[:2]
        
        for line in lines:
            # Project 3D endpoints to 2D
            start_3d = np.array([line['start']], dtype=np.float32)
            end_3d = np.array([line['end']], dtype=np.float32)
            
            start_2d, _ = cv2.projectPoints(
                start_3d,
                camera_params['rvec'],
                camera_params['tvec'],
                camera_params['camera_matrix'],
                camera_params.get('dist_coeffs')
            )
            
            end_2d, _ = cv2.projectPoints(
                end_3d,
                camera_params['rvec'],
                camera_params['tvec'],
                camera_params['camera_matrix'],
                camera_params.get('dist_coeffs')
            )
            
            pt1 = tuple(start_2d[0][0].astype(int))
            pt2 = tuple(end_2d[0][0].astype(int))
            
            # Check if line is visible
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or \
               (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(overlay, pt1, pt2,
                        line['color'], line['thickness'], cv2.LINE_AA)
        
        # Blend with original
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result


class PlayerProjector:
    """Project player positions from tracking data to image"""
    
    def project_players_3d(self, tracking_df, play_id, step, camera_params,
                          player_height=2.0):
        """
        Project players to image using 3D calibration
        
        Args:
            tracking_df: Tracking dataframe
            play_id: Play ID
            step: Frame step
            camera_params: Camera calibration
            player_height: Height in yards to project (torso level)
        
        Returns:
            List of projected player positions
        """
        frame_data = tracking_df[
            (tracking_df['play_id'] == play_id) &
            (tracking_df['step'] == step)
        ]
        
        if len(frame_data) == 0:
            return []
        
        projected = []
        
        for _, player in frame_data.iterrows():
            # 3D position: field x, y, and height z
            pos_3d = np.array([[
                player['x_position'],
                player['y_position'],
                player_height  # Project at player height, not ground
            ]], dtype=np.float32)
            
            # Project to 2D
            pos_2d, _ = cv2.projectPoints(
                pos_3d,
                camera_params['rvec'],
                camera_params['tvec'],
                camera_params['camera_matrix'],
                camera_params.get('dist_coeffs')
            )
            
            x, y = pos_2d[0][0].astype(int)
            
            projected.append({
                'player_id': player['nfl_player_id'],
                'x': x,
                'y': y,
                'jersey': player.get('jersey_number', '?'),
                'team': player.get('team', 'unknown'),
                'field_x': player['x_position'],
                'field_y': player['y_position'],
                'field_z': player_height
            })
        
        return projected
    
    def draw_players(self, image, projected_players):
        """Draw players on image"""
        result = image.copy()
        
        for player in projected_players:
            x, y = player['x'], player['y']
            
            # Skip if outside image
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
            
            # Color by team
            team_str = str(player['team']).lower()
            if 'home' in team_str:
                color = (255, 100, 100)  # Light blue
            else:
                color = (100, 100, 255)  # Light red
            
            # Draw circle
            cv2.circle(result, (x, y), 18, color, -1)
            cv2.circle(result, (x, y), 18, (255, 255, 255), 2)
            
            # Jersey number
            jersey = player['jersey']
            if pd.notna(jersey) and jersey != '?':
                text = str(int(jersey))
            else:
                text = '?'
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            
            cv2.putText(result, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result


def interactive_calibration_tool():
    """
    Interactive tool to manually select calibration points
    """
    print("\n" + "="*70)
    print("INTERACTIVE 3D CAMERA CALIBRATION")
    print("="*70)
    
    print("\nThis tool helps you calibrate your camera by:")
    print("1. Showing you field calibration points to identify")
    print("2. Letting you click on them in the image")
    print("3. Computing camera parameters from correspondences")
    
    # Get inputs
    image_path = "alignment_check_output/first_frame_58221_001269_All29.jpg"
    csv_path = "nfl-player-contact-detection/train_player_tracking.csv"
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"\nImage size: {w} x {h}")
    
    # Estimate visible yard range from tracking
    try:
        df = pd.read_csv(csv_path)
        play_id = int(input("Play ID (for estimating visible field): ").strip())
        step = int(input("Step: ").strip())
        
        frame_data = df[(df['play_id'] == play_id) & (df['step'] == step)]
        
        if len(frame_data) > 0:
            x_min = frame_data['x_position'].min()
            x_max = frame_data['x_position'].max()
            print(f"\nEstimated visible yards: {x_min:.0f} to {x_max:.0f}")
            visible_range = (max(0, x_min - 10), min(120, x_max + 10))
        else:
            print("\nNo tracking data found, using default range")
            visible_range = (10, 110)
    except:
        visible_range = (10, 110)
    
    # Get calibration points
    calibrator = WorldPose3DCalibrator()
    calib_points = calibrator.get_field_calibration_points(visible_range)
    
    print(f"\n{len(calib_points)} calibration points available")
    print("\nYou need to identify at least 6 points (more is better)")
    print("Recommended: Yard line intersections with sidelines")
    
    print("\nAvailable point types:")
    point_types = set(p['type'] for p in calib_points)
    for pt in point_types:
        count = len([p for p in calib_points if p['type'] == pt])
        print(f"  - {pt}: {count} points")
    
    # Instructions for manual point selection
    print("\n" + "="*70)
    print("MANUAL POINT SELECTION INSTRUCTIONS")
    print("="*70)
    print("\n1. Open the image in an image viewer or editor")
    print("2. For each calibration point, note its pixel coordinates (x, y)")
    print("3. Enter the correspondences below")
    print("\nExample points to identify:")
    print("  - 20yd_near_sideline: Where 20-yard line meets near sideline")
    print("  - 30yd_far_sideline: Where 30-yard line meets far sideline")
    print("  - 40yd_left_hash: Where 40-yard line meets left hash mark")
    
    # Collect correspondences
    points_3d = []
    points_2d = []
    
    print("\n" + "="*70)
    print("Enter point correspondences (type 'done' when finished)")
    print("="*70)
    
    while True:
        print(f"\n{len(points_3d)} points entered so far")
        
        if len(points_3d) >= 6:
            done = input("Add another point? (y/n): ").strip().lower()
            if done == 'n':
                break
        
        # Show available points
        print("\nAvailable calibration points:")
        for i, pt in enumerate(calib_points[:20]):  # Show first 20
            print(f"  {i}: {pt['name']} -> {pt['coords_3d']}")
        
        choice = input("\nEnter point number (or 'done'): ").strip()
        
        if choice.lower() == 'done':
            if len(points_3d) < 6:
                print("Need at least 6 points!")
                continue
            break
        
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(calib_points):
                print("Invalid index!")
                continue
            
            selected_point = calib_points[idx]
            
            # Get image coordinates
            img_x = int(input(f"  Image X coordinate for {selected_point['name']}: ").strip())
            img_y = int(input(f"  Image Y coordinate for {selected_point['name']}: ").strip())
            
            points_3d.append(selected_point['coords_3d'])
            points_2d.append([img_x, img_y])
            
            print(f"  ✓ Added: {selected_point['name']}")
            
        except ValueError:
            print("Invalid input!")
            continue
    
    # Perform calibration
    print(f"\n{len(points_3d)} correspondences collected")
    print("Computing calibration...")
    
    camera_params = calibrator.calibrate_from_correspondences(
        points_3d, points_2d, (w, h)
    )
    
    if camera_params is None:
        print("Calibration failed!")
        return
    
    # Save calibration
    output_dir = Path('./worldpose_calibration')
    output_dir.mkdir(exist_ok=True)
    
    calibrator.save_calibration(
        camera_params,
        output_dir / 'camera_calibration.json'
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Draw field lines
    renderer = FieldLineRenderer()
    img_with_lines = renderer.draw_field_lines(img, camera_params, alpha=0.5)
    cv2.imwrite(str(output_dir / 'field_lines_3d.jpg'), img_with_lines)
    print(f"  ✓ Saved: field_lines_3d.jpg")
    
    # Draw players if tracking data available
    if len(frame_data) > 0:
        projector = PlayerProjector()
        projected = projector.project_players_3d(
            df, play_id, step, camera_params, player_height=2.0
        )
        
        img_with_players = projector.draw_players(img_with_lines, projected)
        cv2.imwrite(str(output_dir / 'calibrated_overlay.jpg'), img_with_players)
        print(f"  ✓ Saved: calibrated_overlay.jpg")
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - camera_calibration.json (save this for future use!)")
    print("  - field_lines_3d.jpg")
    print("  - calibrated_overlay.jpg")
    print("\nYou can now use this calibration for all frames from this camera!")
    print("="*70)


if __name__ == "__main__":
    interactive_calibration_tool()