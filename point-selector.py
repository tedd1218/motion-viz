"""
Interactive Point Selection Tool
Click on image to get pixel coordinates for calibration
"""

import cv2
import numpy as np
from pathlib import Path

class PointSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.points = []
        self.point_names = []
        self.window_name = "Point Selection - Click on calibration points"
        
        # Get display size
        screen_height = 1080
        screen_width = 1920
        
        # Resize if image is too large
        h, w = self.image.shape[:2]
        if h > screen_height or w > screen_width:
            scale = min(screen_width / w, screen_height / h) * 0.9
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.display_scale = scale
        else:
            new_w, new_h = w, h
            self.display_scale = 1.0
        
        self.display_size = (new_w, new_h)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original image size
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)
            
            # Get point name
            print(f"\n✓ Point {len(self.points) + 1} clicked: ({orig_x}, {orig_y})")
            point_name = input("  Enter name for this point (e.g., '20yd_near_sideline'): ").strip()
            
            if point_name:
                self.points.append((orig_x, orig_y))
                self.point_names.append(point_name)
                
                # Draw marker on display image
                display_x = int(orig_x * self.display_scale)
                display_y = int(orig_y * self.display_scale)
                
                # Draw circle and label
                color = (0, 255, 0)  # Green
                cv2.circle(self.display_image, (display_x, display_y), 5, color, -1)
                cv2.circle(self.display_image, (display_x, display_y), 8, (255, 255, 255), 2)
                
                # Draw label
                label = f"{len(self.points)}"
                cv2.putText(self.display_image, label, 
                           (display_x + 10, display_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imshow(self.window_name, self.display_image)
                
                print(f"  Added as point #{len(self.points)}: {point_name}")
    
    def select_points(self):
        """Interactive point selection"""
        print("\n" + "="*70)
        print("INTERACTIVE POINT SELECTION")
        print("="*70)
        print("\nInstructions:")
        print("  1. Click on calibration points in the image")
        print("  2. Enter a name for each point when prompted")
        print("  3. Press 'q' when done (need at least 6 points)")
        print("  4. Press 'u' to undo last point")
        print("  5. Press 'r' to restart")
        print("\nGood calibration points:")
        print("  - Yard line intersections with sidelines")
        print("  - Yard line intersections with hash marks")
        print("  - Goal line corners")
        print("\nExample names:")
        print("  - 20yd_near_sideline")
        print("  - 30yd_far_sideline")
        print("  - 40yd_left_hash")
        print("="*70)
        
        # Create resized display image
        self.display_image = cv2.resize(self.image, self.display_size)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions on image
        instructions = [
            "LEFT CLICK: Select point",
            "Q: Quit (need 6+ points)",
            "U: Undo last point",
            "R: Restart"
        ]
        
        y_offset = 30
        for text in instructions:
            cv2.putText(self.display_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        
        cv2.imshow(self.window_name, self.display_image)
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if len(self.points) < 6:
                    print(f"\n⚠ Need at least 6 points (currently have {len(self.points)})")
                else:
                    break
            
            elif key == ord('u'):
                if len(self.points) > 0:
                    removed_point = self.points.pop()
                    removed_name = self.point_names.pop()
                    print(f"\n↶ Undid point: {removed_name} at {removed_point}")
                    
                    # Redraw image
                    self.display_image = cv2.resize(self.image, self.display_size)
                    
                    # Redraw remaining points
                    for i, (px, py) in enumerate(self.points):
                        display_x = int(px * self.display_scale)
                        display_y = int(py * self.display_scale)
                        cv2.circle(self.display_image, (display_x, display_y), 5, (0, 255, 0), -1)
                        cv2.circle(self.display_image, (display_x, display_y), 8, (255, 255, 255), 2)
                        label = f"{i+1}"
                        cv2.putText(self.display_image, label,
                                   (display_x + 10, display_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(self.window_name, self.display_image)
            
            elif key == ord('r'):
                print("\n🔄 Restarting - all points cleared")
                self.points = []
                self.point_names = []
                self.display_image = cv2.resize(self.image, self.display_size)
                cv2.imshow(self.window_name, self.display_image)
        
        cv2.destroyAllWindows()
        
        print(f"\n✓ Selection complete! {len(self.points)} points collected")
        return self.points, self.point_names
    
    def save_points(self, output_path):
        """Save selected points to file"""
        with open(output_path, 'w') as f:
            f.write("# Point correspondences for camera calibration\n")
            f.write("# Format: point_name, image_x, image_y\n\n")
            
            for name, (x, y) in zip(self.point_names, self.points):
                f.write(f"{name}, {x}, {y}\n")
        
        print(f"✓ Points saved to {output_path}")
    
    def visualize_selections(self, output_path):
        """Save image with marked points"""
        vis_image = self.image.copy()
        
        for i, ((x, y), name) in enumerate(zip(self.points, self.point_names)):
            # Draw marker
            cv2.circle(vis_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(vis_image, (x, y), 12, (255, 255, 255), 2)
            
            # Draw label
            label = f"{i+1}: {name}"
            cv2.putText(vis_image, label, (x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"✓ Visualization saved to {output_path}")


def main():
    print("\n" + "="*70)
    print("CALIBRATION POINT SELECTION TOOL")
    print("="*70)
    
    image_path = "alignment_check_output/first_frame_58221_001269_All29.jpg"
    
    try:
        selector = PointSelector(image_path)
        points, names = selector.select_points()
        
        if len(points) >= 6:
            # Save points
            output_dir = Path('./calibration_points')
            output_dir.mkdir(exist_ok=True)
            
            selector.save_points(output_dir / 'selected_points.txt')
            selector.visualize_selections(output_dir / 'marked_points.jpg')
            
            # Print summary
            print("\n" + "="*70)
            print("SELECTED POINTS SUMMARY")
            print("="*70)
            for i, (name, (x, y)) in enumerate(zip(names, points)):
                print(f"{i+1}. {name}: ({x}, {y})")
            
            print("\n" + "="*70)
            print("FILES SAVED:")
            print("="*70)
            print(f"  - {output_dir / 'selected_points.txt'}")
            print(f"  - {output_dir / 'marked_points.jpg'}")
            print("\nYou can now use these points for calibration!")
            print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()