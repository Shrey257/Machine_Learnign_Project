import cv2
import numpy as np

class BallTracker:
    """Class for tracking the ball in football videos"""
    
    def __init__(self, max_disappeared=15):
        """Initialize the ball tracker"""
        self.tracked = False
        self.current_position = None
        self.bbox = None
        self.trajectory = []
        self.disappeared = 0
        self.max_disappeared = max_disappeared
    
    def update(self, centroid=None, bbox=None):
        """Update the ball's position"""
        if centroid is None:
            # No detection, increment disappeared counter
            self.disappeared += 1
            
            # If the ball has disappeared for too long, mark as not tracked
            if self.disappeared > self.max_disappeared:
                self.tracked = False
                self.current_position = None
                self.bbox = None
        else:
            # Update with new position
            self.tracked = True
            self.current_position = centroid
            self.bbox = bbox
            self.disappeared = 0
            
            # Add to trajectory
            self.trajectory.append(centroid)
            
            # Limit trajectory length to prevent memory issues
            max_trajectory_length = 90  # ~3 seconds at 30fps
            if len(self.trajectory) > max_trajectory_length:
                self.trajectory = self.trajectory[-max_trajectory_length:]
        
        return self.tracked
    
    def predict_position(self, num_frames=1):
        """Predict the ball's position in future frames based on current trajectory"""
        if not self.tracked or len(self.trajectory) < 2:
            return None
        
        # Get the last two positions
        last_pos = self.trajectory[-1]
        prev_pos = self.trajectory[-2]
        
        # Calculate velocity
        vx = last_pos[0] - prev_pos[0]
        vy = last_pos[1] - prev_pos[1]
        
        # Predict position
        pred_x = last_pos[0] + vx * num_frames
        pred_y = last_pos[1] + vy * num_frames
        
        return (int(pred_x), int(pred_y))
    
    def calculate_speed(self, pixels_to_meters, fps):
        """Calculate the ball's speed in meters per second"""
        if not self.tracked or len(self.trajectory) < 2:
            return 0.0
        
        # Only use the last few points to calculate current speed
        window = min(5, len(self.trajectory) - 1)
        distances = []
        
        for i in range(1, window + 1):
            p1 = self.trajectory[-i-1]
            p2 = self.trajectory[-i]
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            distances.append(dist)
        
        # Average distance per frame
        avg_dist = np.mean(distances)
        
        # Convert to meters per second
        speed = avg_dist * pixels_to_meters * fps
        
        return speed
    
    def get_direction_vector(self):
        """Get the ball's current direction vector"""
        if not self.tracked or len(self.trajectory) < 2:
            return (0, 0)
        
        # Get the last two positions
        last_pos = self.trajectory[-1]
        prev_pos = self.trajectory[-2]
        
        # Calculate direction vector
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        
        # Normalize
        mag = np.sqrt(dx**2 + dy**2)
        if mag > 0:
            dx /= mag
            dy /= mag
        
        return (dx, dy)
    
    def is_in_possession(self, player_positions, proximity_threshold=50):
        """Determine if the ball is in possession by any player"""
        if not self.tracked:
            return None
        
        # Find the closest player to the ball
        min_dist = float('inf')
        closest_player = None
        
        for player_id, position in player_positions.items():
            # Calculate distance between ball and player
            dx = self.current_position[0] - position[0]
            dy = self.current_position[1] - position[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = player_id
        
        # If a player is close enough, return the player ID
        if min_dist <= proximity_threshold:
            return closest_player
        
        return None 