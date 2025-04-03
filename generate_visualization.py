import cv2
import numpy as np
import time
import os
from player_tracker import PlayerTracker
from report_generator import ReportGenerator
import config

def create_demo_visualization():
    """
    Generate a visualization similar to the screenshot in the example, then create a report
    """
    # Make sure output directories exist
    os.makedirs(os.path.join('output', 'videos'), exist_ok=True)
    os.makedirs(os.path.join('output', 'reports'), exist_ok=True)
    
    # Output paths
    output_video_path = os.path.join('output', 'videos', 'demo_output.mp4')
    report_path = os.path.join('output', 'reports', 'player_stats_report.html')
    
    # Create a blank football field image (green with white lines)
    field_width, field_height = 1280, 720
    field_image = np.ones((field_height, field_width, 3), dtype=np.uint8) * np.array([80, 160, 80], dtype=np.uint8)
    
    # Draw field markings
    cv2.circle(field_image, (field_width // 2, field_height // 2), 91, (255, 255, 255), 2)  # Center circle
    cv2.line(field_image, (field_width // 2, 0), (field_width // 2, field_height), (255, 255, 255), 2)  # Halfway line
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (field_width, field_height))
    
    # Create player tracker
    player_tracker = PlayerTracker()
    
    # Create report generator
    report_gen = ReportGenerator(report_path)
    
    # Define team colors for visualization
    team_a_color = (0, 0, 255)  # Red (BGR) for Team A (white jerseys)
    team_b_color = (255, 0, 0)  # Blue (BGR) for Team B (green jerseys)
    
    # Create simulated player positions
    # Team A players (white jerseys, red boxes)
    team_a_positions = [
        (100, 200),   # Player 1
        (240, 350),   # Player 3
        (380, 180),   # Player 5
        (620, 520),   # Player 7
        (450, 650),   # Player 9
        (750, 320),   # Player 11
        (570, 150),   # Player 13
        (320, 450),   # Player 19
        (620, 290),   # Player 25
        (200, 600),   # Player 27
        (800, 150),   # Player 29
    ]
    
    # Team B players (green jerseys, blue boxes)
    team_b_positions = [
        (950, 520),   # Player 0
        (780, 400),   # Player 2
        (850, 200),   # Player 6
        (700, 600),   # Player 8
        (450, 300),   # Player 14
        (540, 400),   # Player 15
        (650, 350),   # Player 17
        (500, 500),   # Player 20
        (850, 650),   # Player 26
    ]
    
    # Ball position
    ball_position = (500, 390)
    
    # Register players with the tracker
    # Team A (white jerseys)
    for i, pos in enumerate(team_a_positions):
        player_id = i * 2 + 1  # Odd numbers for Team A
        # Register player with the tracker
        player_tracker.register(pos, (pos[0]-20, pos[1]-40, 40, 80), field_image)
        # Force team assignment
        player_tracker.teams[player_id] = "team_a"
        player_tracker.player_teams[player_id] = "A"
        # Generate random velocities and distances
        player_tracker.velocities[player_id] = np.random.uniform(2.0, 12.0)  # km/h
        player_tracker.total_distances[player_id] = np.random.uniform(5.0, 30.0)  # meters
    
    # Team B (green jerseys)
    for i, pos in enumerate(team_b_positions):
        player_id = i * 2  # Even numbers for Team B
        # Register player with the tracker
        player_tracker.register(pos, (pos[0]-20, pos[1]-40, 40, 80), field_image)
        # Force team assignment
        player_tracker.teams[player_id] = "team_b"
        player_tracker.player_teams[player_id] = "B"
        # Generate random velocities and distances
        player_tracker.velocities[player_id] = np.random.uniform(2.0, 14.0)  # km/h
        player_tracker.total_distances[player_id] = np.random.uniform(5.0, 35.0)  # meters
    
    # Ball (ID 0)
    player_tracker.objects[0] = {
        "centroid": ball_position,
        "centroids": [ball_position],
        "rect": (ball_position[0]-10, ball_position[1]-10, 20, 20),
        "bbox": (ball_position[0]-10, ball_position[1]-10, 20, 20),
        "disappeared": 0,
        "confidence": 1.0,
        "class": 32,  # Ball class
        "timestamps": [time.time()],
        "positions": [ball_position],
        "velocity": 0,
        "distance": 0,
        "team": None
    }
    
    # Generate visualization for the screenshot
    frame = field_image.copy()
    
    # Draw players and ball
    for object_id, obj in player_tracker.objects.items():
        if object_id == 0:  # Ball
            # Draw the ball as a yellow circle
            ball_color = (0, 215, 255)  # Yellow in BGR
            cv2.circle(frame, obj["centroid"], 8, ball_color, -1)
            continue
            
        # Get the current position and bounding box
        if "rect" in obj:
            x, y, w, h = obj["rect"]
        else:
            centroid = obj["centroid"]
            w, h = 40, 80  # Default player size
            x, y = centroid[0] - w//2, centroid[1] - h//2
            
        # Determine the team color
        if player_tracker.player_teams.get(object_id) == "A":
            color = team_a_color  # Red for white jerseys (Team A)
        else:
            color = team_b_color  # Blue for green jerseys (Team B)
            
        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Get velocity and distance
        velocity = player_tracker.velocities.get(object_id, 0)
        distance = player_tracker.total_distances.get(object_id, 0)
        
        # Format text with player ID, velocity, and distance
        id_text = f"ID: {object_id}"
        vel_text = f"V: {velocity:.1f} km/h"
        dist_text = f"D: {distance:.1f} m"
        
        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Draw player ID at the top
        cv2.rectangle(frame, 
                    (x, y - 20), 
                    (x + 80, y), 
                    color, -1)
        cv2.putText(frame, id_text, 
                  (x + 5, y - 5), 
                  font, font_scale, (255, 255, 255), thickness)
        
        # Draw velocity in the middle
        cv2.rectangle(frame, 
                    (x, y + h), 
                    (x + 80, y + h + 20), 
                    color, -1)
        cv2.putText(frame, vel_text, 
                  (x + 5, y + h + 15), 
                  font, font_scale, (255, 255, 255), thickness)
        
        # Draw distance at the bottom
        cv2.rectangle(frame, 
                    (x, y + h + 20), 
                    (x + 80, y + h + 40), 
                    color, -1)
        cv2.putText(frame, dist_text, 
                  (x + 5, y + h + 35), 
                  font, font_scale, (255, 255, 255), thickness)
    
    # Add a player count at the top
    player_text = f"Players: {len(player_tracker.objects) - 1}"  # Subtract 1 for the ball
    cv2.putText(frame, player_text, (20, 40), font, 1, (255, 255, 255), 2)
    
    # Add stadium background at the top
    stadium_background = np.zeros((80, field_width, 3), dtype=np.uint8)
    stadium_background[:, :] = (50, 50, 50)  # Dark gray
    
    # Add red advertisement band at the bottom like in the screenshot
    ad_band_height = 50
    ad_band = np.zeros((ad_band_height, field_width, 3), dtype=np.uint8)
    ad_band[:, :] = (0, 0, 200)  # Red
    
    # Create a composite image with the stadium background at top
    composite = np.vstack([stadium_background, frame, ad_band])
    
    # Resize to original dimensions if needed
    if composite.shape[0] != field_height:
        composite = cv2.resize(composite, (field_width, field_height))
    
    # Write the frame to video
    out.write(frame)  # Use the original frame without the extra bands for video
    
    # Save a screenshot that looks like the example
    cv2.imwrite(os.path.join('output', 'videos', 'demo_screenshot.jpg'), composite)
    
    # Add the frame data to the report generator
    report_gen.update_frame_data(
        1,  # Frame number
        player_tracker.player_teams,
        player_tracker.velocities,
        player_tracker.total_distances
    )
    
    # Release the video writer
    out.release()
    
    # Generate report
    report_gen.generate_report()
    
    print(f"Demo visualization created at: {output_video_path}")
    print(f"Screenshot saved at: {os.path.join('output', 'videos', 'demo_screenshot.jpg')}")
    print(f"Player statistics report generated at: {report_path}")
    
    # Return paths to results
    return output_video_path, report_path

if __name__ == "__main__":
    video_path, report_path = create_demo_visualization()
    
    # Try to open the video and report if on Windows
    import platform
    if platform.system() == "Windows":
        import os
        os.system(f'start "{os.path.join("output", "videos", "demo_screenshot.jpg")}"')
        os.system(f'start "{report_path}"') 