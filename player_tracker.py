import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
import config
import time
from scipy.optimize import linear_sum_assignment

class PlayerTracker:
    def __init__(self):
        # Dictionary to store active objects being tracked
        self.objects = OrderedDict()
        
        # Dictionary to store centroids history for trajectory visualization
        self.trajectories = OrderedDict()
        
        # Dictionary to store disappearance count
        self.disappeared = OrderedDict()
        
        # Dictionary to store team assignments
        self.teams = OrderedDict()
        self.player_teams = OrderedDict()  # Alias for teams to maintain compatibility with main.py
        
        # Dictionary to store jersey color histograms for each player
        self.player_color_features = OrderedDict()
        
        # Dictionary to store player color histograms for each player
        self.player_color_histograms = OrderedDict()
        
        # Alias for player_color_features for backward compatibility
        self.color_features = self.player_color_features
        
        # Dictionary to track frames each player has been tracked
        self.frames_tracked = OrderedDict()
        
        # Set team colors as None initially - will be automatically determined
        self.team_colors_determined = False
        self.team_a_color = None
        self.team_b_color = None
        
        # Create color bins for histogram
        self.color_bins = (40, 40)  # Increased for better color differentiation
        
        # Sampling ratio for color extraction
        self.color_sample_ratio = 0.5  # Use 50% of pixels for color extraction
        
        # For velocity and distance calculation
        self.prev_positions = {}  # Previous positions for velocity calculation
        self.position_history = {}  # Complete history of positions
        self.velocities = {}  # Current velocity of each player
        self.total_distances = {}  # Total distance traveled by each player
        
        # For appearance-based matching
        self.appearance_embeddings = {}  # Store appearance features for each player
        
        # Increase MAX_DISAPPEARED for better tracking persistence
        self.max_disappeared = config.MAX_DISAPPEARED if hasattr(config, 'MAX_DISAPPEARED') else 30
        self.max_distance = config.MAX_DISTANCE if hasattr(config, 'MAX_DISTANCE') else 100
        
        # Next object ID
        self.next_object_id = 0

        # Team color dominant values
        self.team_a_dominant_colors = []
        self.team_b_dominant_colors = []
        
        # Color samples for team clustering
        self.color_samples = []
        self.player_ids_for_colors = []
        
        # Tracking statistics
        self.tracking_start_times = {}
        
        # Counter for team classification attempts
        self.team_class_attempts = 0
        self.min_players_for_classification = config.MIN_PLAYERS_FOR_TEAM_CLASSIFICATION if hasattr(config, 'MIN_PLAYERS_FOR_TEAM_CLASSIFICATION') else 4
        
        # Store last frame for color extraction
        self.last_frame = None

    def register(self, centroid, bbox, frame):
        """Register a new object with the next available ID"""
        # Store the centroid and initialize the disappeared counter
        self.objects[self.next_object_id] = {
            "centroids": [centroid],
            "centroid": centroid,
            "rect": bbox,
            "bbox": bbox,  # Add bbox for compatibility
            "disappeared": 0,
            "confidence": 1.0,
            "class": 0,
            "timestamps": [time.time()],
            "positions": [centroid],
            "velocity": 0,
            "distance": 0,
            "team": None
        }
        
        # Initialize trajectory
        self.trajectories[self.next_object_id] = [centroid]
        
        # Initialize position history
        self.position_history[self.next_object_id] = [centroid]
        
        # Initialize metrics
        self.velocities[self.next_object_id] = 0.0
        self.total_distances[self.next_object_id] = 0.0
        self.tracking_start_times[self.next_object_id] = time.time()
        self.frames_tracked[self.next_object_id] = 1
        
        # Extract appearance features
        self.extract_color_features(self.next_object_id, frame)
        
        # Determine team if possible
        if self.team_colors_determined:
            team = self.determine_player_team(self.next_object_id)
            self.teams[self.next_object_id] = team
            
            # Update player_teams mapping
            if team == "team_a":
                self.player_teams[self.next_object_id] = "A"
            elif team == "team_b":
                self.player_teams[self.next_object_id] = "B"
            else:
                self.player_teams[self.next_object_id] = "Unknown"
        else:
            self.teams[self.next_object_id] = "unknown"
            self.player_teams[self.next_object_id] = "Unknown"
        
        # Increment ID counter
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an object that is no longer being tracked"""
        # Delete the object from our dictionaries
        del self.objects[object_id]
        del self.trajectories[object_id]
        del self.disappeared[object_id]
        
        # Delete from team assignments if assigned
        if object_id in self.teams:
            del self.teams[object_id]
        
        # Delete color features if extracted
        if object_id in self.player_color_features:
            del self.player_color_features[object_id]
        if object_id in self.player_color_histograms:
            del self.player_color_histograms[object_id]
        if object_id in self.total_distances:
            del self.total_distances[object_id]
        if object_id in self.velocities:
            del self.velocities[object_id]
        if object_id in self.appearance_embeddings:
            del self.appearance_embeddings[object_id]
        if object_id in self.position_history:
            del self.position_history[object_id]
        if object_id in self.tracking_start_times:
            del self.tracking_start_times[object_id]

    def update(self, detections, frame):
        """Update the tracker with new detections and enforce balanced team assignment"""
        # Store the frame for later use
        self.last_frame = frame
        
        # Register new centroids with detections provided
        objects = self.register_new_detections(detections, frame)
        
        # If no detections or no objects being tracked, return empty objects
        if len(objects) == 0:
            return objects
            
        # Calculate centroids for all current objects
        centroids = np.array([self.objects[object_id]['centroid'] for object_id in self.objects])
        rects = [self.objects[object_id]['rect'] for object_id in self.objects]
        
        # Calculate centroids for all new detections
        new_centroids = np.array([detection[0:2] for detection in detections])
        new_rects = [detection[2:6] for detection in detections]
        
        # Calculate distances between existing and new centroids
        D = dist.cdist(centroids, new_centroids)
        
        # Calculate IoU between existing and new bounding boxes for better matching
        iou_matrix = np.zeros((len(rects), len(new_rects)))
        for i, rect1 in enumerate(rects):
            for j, rect2 in enumerate(new_rects):
                iou_matrix[i, j] = self.bbox_iou(rect1, rect2)
        
        # Check for appearance similarity based on jersey color
        appearance_cost = np.zeros((len(self.objects), len(detections)))
        for i, object_id in enumerate(self.objects):
            if object_id in self.player_color_features:
                for j, _ in enumerate(detections):
                    # Extract color feature from detection
                    x, y, w, h = new_rects[j]
                    detection_color = self.extract_jersey_color(frame, (x, y, w, h))
                    
                    if detection_color is not None and object_id in self.player_color_features:
                        # Calculate color similarity
                        color_diff = np.linalg.norm(detection_color - self.player_color_features[object_id])
                        appearance_cost[i, j] = color_diff
                    else:
                        appearance_cost[i, j] = 1000  # High cost for missing color
        
        # Normalize appearance cost
        if appearance_cost.size > 0:
            appearance_cost = appearance_cost / (np.max(appearance_cost) + 1e-6)
        
        # Create combined cost matrix
        if hasattr(config, 'APPEARANCE_WEIGHT') and config.APPEARANCE_WEIGHT:
            combined_cost = (1 - config.APPEARANCE_WEIGHT) * D + config.APPEARANCE_WEIGHT * appearance_cost
        else:
            combined_cost = D
        
        # Apply IoU bonus for overlapping boxes
        iou_bonus = 1 - iou_matrix  # Convert IoU to cost (lower is better)
        combined_cost = 0.7 * combined_cost + 0.3 * iou_bonus
        
        # Hungarian algorithm to find optimal matching
        row_ind, col_ind = linear_sum_assignment(combined_cost)
        
        # Create dictionaries to store used rows and columns
        used_rows = set()
        used_cols = set()
        
        # Go through the matches to update object positions
        matched_objects = {}
        for (row, col) in zip(row_ind, col_ind):
            if combined_cost[row, col] < self.max_distance:  # Only consider valid matches
                object_id = list(self.objects.keys())[row]
                detection = detections[col]
                
                # Add to lists of used rows and columns
                used_rows.add(row)
                used_cols.add(col)
                
                # Extract coordinates 
                x, y, width, height, confidence, class_id = detection
                centroid = (x, y)
                rect = (int(x - width/2), int(y - height/2), width, height)
                
                # Update the object
                matched_objects[object_id] = {
                    'detection': detection,
                    'centroid': centroid,
                    'rect': rect,
                    'disappeared': 0
                }
                
                # Update trajectory
                if object_id in self.trajectories:
                    self.trajectories[object_id].append(centroid)
                else:
                    self.trajectories[object_id] = [centroid]
                    
                # Update frames tracked
                if object_id in self.frames_tracked:
                    self.frames_tracked[object_id] += 1
                else:
                    self.frames_tracked[object_id] = 1
                    
                # Update velocity
                if object_id in self.trajectories and len(self.trajectories[object_id]) >= 2:
                    self.calculate_velocity(object_id)
        
        # Extract the set of unused row and column indexes
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        
        # Mark objects with no matches as disappeared
        for row in unused_rows:
            object_id = list(self.objects.keys())[row]
            self.objects[object_id]['disappeared'] += 1
            
            # Keep tracking if within maximum disappearance threshold
            if self.objects[object_id]['disappeared'] <= self.max_disappeared:
                matched_objects[object_id] = self.objects[object_id]
                
        # Register each new detection as a new object
        for col in unused_cols:
            detection = detections[col]
            self.register_object(detection, matched_objects, frame)
            
        # Update the objects being tracked
        self.objects = matched_objects
        
        # Try to determine team colors if not already done
        if not self.team_colors_determined and \
           len(self.objects) >= config.MIN_PLAYERS_FOR_TEAM_CLASSIFICATION and \
           self.team_class_attempts < 3:  # Limit attempts to prevent constant recalculation
            
            # Try team color determination
            self.determine_team_colors(frame)
            self.team_class_attempts += 1
            
        # Determine teams for all objects if we have team colors
        if self.team_colors_determined:
            for object_id in self.objects:
                if object_id not in self.teams or self.teams[object_id] == "unknown":
                    team = self.determine_player_team(object_id)
                    self.teams[object_id] = team
            
            # Update player_teams mapping to stay in sync with teams
            self.update_player_teams_map()
                    
        # Enforce balanced teams if configured
        if hasattr(config, 'ENFORCE_BALANCED_TEAMS') and config.ENFORCE_BALANCED_TEAMS:
            self.balance_teams()
        
        return self.objects
    
    def extract_color_histogram(self, player_patch):
        """Extract color histogram from player patch for jersey color identification"""
        if player_patch is None or player_patch.size == 0 or player_patch.shape[0] <= 0 or player_patch.shape[1] <= 0:
            return None
        
        # Convert to HSV color space
        try:
            hsv = cv2.cvtColor(player_patch, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return None
        
        # Get patch dimensions
        h, w = player_patch.shape[:2]
        
        # Extract only the upper body (jersey) region
        jersey_roi_ratio = config.JERSEY_ROI_RATIO if hasattr(config, 'JERSEY_ROI_RATIO') else 0.4
        jersey_height = int(h * jersey_roi_ratio)
        if jersey_height <= 0:
            jersey_height = 1
        
        jersey_region = hsv[0:jersey_height, :]
        
        if jersey_region.size == 0:
            return None  # Return None if region is empty
        
        # Create a mask to exclude background (usually green field)
        # Exclude pixels close to the field color in HSV
        mask = cv2.inRange(
            jersey_region,
            np.array([35, 40, 40]),   # Lower bound for field color (usually green)
            np.array([90, 255, 255])  # Upper bound for field color
        )
        # Invert the mask to keep only non-field pixels
        mask = cv2.bitwise_not(mask)
        
        # Check if we have enough non-field pixels
        if cv2.countNonZero(mask) < 20:  # If fewer than 20 non-field pixels, use full image
            mask = np.ones_like(jersey_region[:,:,0], dtype=np.uint8) * 255
        
        # Calculate histogram using the mask
        hist = cv2.calcHist(
            [jersey_region],
            [0, 1],  # Use hue and saturation channels
            mask,
            self.color_bins,  # Number of bins
            [0, 180, 0, 256]  # Ranges for H and S channels
        )
        
        # Normalize the histogram
        try:
            hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        except cv2.error:
            return None
        
        return hist
    
    def extract_jersey_color(self, frame, bbox):
        """
        Extract the most dominant jersey color from a player's bounding box.
        Optimized to identify white and green jerseys with dedicated masks.
        
        Args:
            frame: The current frame
            bbox: The bounding box of the player (x, y, w, h)
            
        Returns:
            The dominant HSV color in the jersey region
        """
        try:
            # Convert bbox coordinates to integers
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Define the jersey region (top half of the bounding box)
            jersey_ratio = 0.4  # Focus on top part of bounding box
            
            jersey_height = int(h * jersey_ratio)
            jersey_y = max(0, y)
            
            # Ensure coordinates are within frame bounds
            if x < 0 or jersey_y < 0 or x + w > frame.shape[1] or jersey_y + jersey_height > frame.shape[0]:
                # Adjust coordinates to be within bounds
                x = max(0, x)
                jersey_y = max(0, jersey_y)
                w = min(frame.shape[1] - x, w)
                jersey_height = min(frame.shape[0] - jersey_y, jersey_height)
                
                # Check if region is too small after adjustments
                if w <= 0 or jersey_height <= 0:
                    return None
            
            # Extract the jersey region
            jersey_roi = frame[jersey_y:jersey_y+jersey_height, x:x+w]
            
            # If ROI is empty, return None
            if jersey_roi.size == 0:
                return None
                
            # Convert to HSV
            hsv_roi = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)
            
            # Define masks for white jerseys (Team A) and green jerseys (Team B)
            # White jersey: very low saturation, high value
            white_jersey_mask = cv2.inRange(
                hsv_roi,
                np.array(config.WHITE_JERSEY_HSV_RANGE_LOW),  # Lower bounds for white
                np.array(config.WHITE_JERSEY_HSV_RANGE_HIGH)  # Upper bounds for white
            )
            
            # Green jersey: green hue range, higher saturation
            green_jersey_mask = cv2.inRange(
                hsv_roi,
                np.array(config.GREEN_JERSEY_HSV_RANGE_LOW),  # Lower bounds for green
                np.array(config.GREEN_JERSEY_HSV_RANGE_HIGH)  # Upper bounds for green
            )
            
            # Count pixels in each mask
            total_pixels = jersey_roi.size / 3  # Divide by 3 for the 3 color channels
            white_pixels = cv2.countNonZero(white_jersey_mask)
            green_pixels = cv2.countNonZero(green_jersey_mask)
            
            # Calculate percentages
            white_percent = white_pixels / total_pixels * 100 if total_pixels > 0 else 0
            green_percent = green_pixels / total_pixels * 100 if total_pixels > 0 else 0
            
            # Debug info
            if config.VERBOSE and (white_percent > 20 or green_percent > 20):
                print(f"Jersey color: white={white_percent:.1f}%, green={green_percent:.1f}%")
            
            # Check if jersey is predominantly white (Team A)
            if white_percent > 25 and white_percent > green_percent:
                return np.array(config.TEAM_A_HSV_COLOR)  # White jersey in HSV
            
            # Check if jersey is predominantly green (Team B)
            if green_percent > 20 and green_percent > white_percent:
                return np.array(config.TEAM_B_HSV_COLOR)  # Green jersey in HSV
            
            # If no clear dominant color, use K-means clustering to find dominant color
            # Reshape for K-means
            pixels = hsv_roi.reshape(-1, 3).astype(np.float32)
            
            # Skip if there are too few pixels
            if pixels.shape[0] < 5:
                return None
                
            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Count pixels in each cluster
            cluster_counts = np.bincount(labels.flatten())
            
            # Get dominant cluster
            dominant_cluster = np.argmax(cluster_counts)
            dominant_color = centers[dominant_cluster]
            
            # Check if this resembles white (low S, high V)
            if dominant_color[1] < 40 and dominant_color[2] > 150:
                return np.array(config.TEAM_A_HSV_COLOR)  # White jersey
                
            # Check if this resembles green (green H range, higher S)
            if 45 <= dominant_color[0] <= 95 and dominant_color[1] > 50:
                return np.array(config.TEAM_B_HSV_COLOR)  # Green jersey
                
            return dominant_color  # Return actual dominant color if not white or green
        except Exception as e:
            print(f"Error in extract_jersey_color: {e}")
            return None
    
    def extract_color_features(self, object_id, frame):
        """
        Extract color features from the bounding box to be used for player re-identification.
        
        Args:
            object_id: The ID of the player
            frame: The current frame
            
        Returns:
            A feature vector containing color histogram information
        """
        try:
            # Get the bounding box
            if "bbox" in self.objects[object_id]:
                x, y, w, h = self.objects[object_id]["bbox"]
            elif "rect" in self.objects[object_id]:
                x, y, w, h = self.objects[object_id]["rect"]
            else:
                return None  # No bounding box available
            
            # Ensure coordinates are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure the bounding box is within the frame
            if x < 0 or y < 0 or x + w >= frame.shape[1] or y + h >= frame.shape[0]:
                x = max(0, x)
                y = max(0, y)
                w = min(frame.shape[1] - x - 1, w)
                h = min(frame.shape[0] - y - 1, h)
                
                # If the adjusted box is too small, return None
                if w <= 0 or h <= 0:
                    return None
            
            # Extract the ROI
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None
            
            # Convert to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Compute histograms for each channel
            h_hist = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv_roi], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv_roi], [2], None, [16], [0, 256])
            
            # Normalize the histograms
            cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Combine the histograms into a single feature vector
            feature_vector = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            
            # Extract dominant jersey color for team classification
            jersey_color = self.extract_jersey_color(frame, (x, y, w, h))
            if jersey_color is not None:
                self.player_color_features[object_id] = jersey_color
                
                # Add to color samples for team clustering
                # Only add samples from players who have been tracked for a while
                min_tracking_time = 1.0  # seconds
                if object_id in self.tracking_start_times:
                    tracking_time = time.time() - self.tracking_start_times[object_id]
                    if tracking_time > min_tracking_time:
                        self.color_samples.append(jersey_color)
                        self.player_ids_for_colors.append(object_id)
            
            # Try to determine team colors if we have enough samples
            if (not self.team_colors_determined and 
                len(self.color_samples) >= self.min_players_for_classification and
                self.team_class_attempts < 3):  # Limit attempts to prevent constant recalculation
                
                self.determine_team_colors(frame)
                self.team_class_attempts += 1
            
            # Assign team to this player
            if self.team_colors_determined and object_id in self.player_color_features:
                self.determine_player_team(object_id)
            
            return feature_vector
        except Exception as e:
            print(f"Error in extract_color_features: {e}")
            return None
    
    def update_velocity_and_distance(self, object_id):
        """Update velocity and distance for a player"""
        if object_id not in self.objects:
            return
        
        # Get current position
        if "centroid" in self.objects[object_id]:
            current_pos = np.array(self.objects[object_id]["centroid"])
        elif "centroids" in self.objects[object_id] and self.objects[object_id]["centroids"]:
            current_pos = np.array(self.objects[object_id]["centroids"][-1])
        else:
            # If no centroid info, try to get from rect or bbox
            if "rect" in self.objects[object_id]:
                x, y, w, h = self.objects[object_id]["rect"]
                current_pos = np.array([x + w/2, y + h/2])
            elif "bbox" in self.objects[object_id]:
                x, y, w, h = self.objects[object_id]["bbox"]
                current_pos = np.array([x + w/2, y + h/2])
            else:
                return  # No position info available
        
        # Initialize position history if not already done
        if object_id not in self.position_history:
            self.position_history[object_id] = [current_pos]
            return
        
        # Check if position has changed significantly to avoid unnecessary updates
        prev_pos = self.position_history[object_id][-1]
        distance_pixels = np.linalg.norm(current_pos - prev_pos)
        
        # Only update if moved more than minimal distance (reduces jitter)
        min_distance = 3.0  # pixels
        if distance_pixels < min_distance:
            return
            
        # Add to position history
        self.position_history[object_id].append(current_pos)
        
        # Keep only the last window_size positions
        window_size = config.VELOCITY_WINDOW if hasattr(config, 'VELOCITY_WINDOW') else 5
        if len(self.position_history[object_id]) > window_size:
            self.position_history[object_id] = self.position_history[object_id][-window_size:]
        
        # Calculate velocity (pixels per frame)
        if len(self.position_history[object_id]) >= 2:
            # Get previous position
            prev_pos = self.position_history[object_id][-2]
            
            # Calculate distance moved
            distance_pixels = np.linalg.norm(current_pos - prev_pos)
            
            # Update total distance
            pixels_to_meters = config.PIXELS_TO_METERS if hasattr(config, 'PIXELS_TO_METERS') else 0.1
            self.total_distances[object_id] = self.total_distances.get(object_id, 0.0) + distance_pixels * pixels_to_meters
            
            # Calculate velocity
            fps = config.FPS if hasattr(config, 'FPS') else 30.0
            velocity_mps = distance_pixels * pixels_to_meters * fps
            
            # Apply maximum realistic speed filter
            max_speed = config.MAX_REALISTIC_SPEED if hasattr(config, 'MAX_REALISTIC_SPEED') else 12.0
            if velocity_mps > max_speed:
                velocity_mps = self.velocities.get(object_id, 0.0)  # Keep previous value if unrealistic
            
            # Calculate velocity in km/h for display
            velocity_kmh = velocity_mps * 3.6
            
            # Store velocity
            self.velocities[object_id] = velocity_kmh

        # Update player_teams mapping to match teams
        self.update_player_teams_map()

    def update_player_teams_map(self):
        """Update player_teams mapping to match teams for compatibility"""
        for object_id, team in self.teams.items():
            # Convert team_a/team_b format to A/B format
            if team == "team_a":
                self.player_teams[object_id] = "A"
            elif team == "team_b":
                self.player_teams[object_id] = "B"
            else:
                self.player_teams[object_id] = "Unknown"

    def determine_team_colors(self, frame):
        """
        Determine team colors specifically optimized for white vs green jerseys
        """
        # If using fixed team colors from config
        if config.USE_FIXED_TEAM_COLORS:
            self.team_a_dominant_colors = [config.TEAM_A_HSV_COLOR]  # White jerseys
            self.team_b_dominant_colors = [config.TEAM_B_HSV_COLOR]  # Green jerseys
            self.team_colors_determined = True
            print("Using fixed team colors: Team A (White), Team B (Green)")
            return True
            
        # Check if we have enough color samples
        min_samples = config.MIN_PLAYERS_FOR_TEAM_CLASSIFICATION
        
        # Store jersey colors of detected players
        team_a_colors = []
        team_b_colors = []
        
        # If we don't have enough color samples but there are players
        if len(self.color_samples) < min_samples and len(self.color_samples) > 0:
            # Force team assignment with fixed colors if needed
            if config.FORCE_TEAM_ASSIGNMENT:
                self.team_a_dominant_colors = [[0, 0, 200]]  # White jersey in HSV
                self.team_b_dominant_colors = [[60, 180, 180]]  # Green jersey in HSV
                self.team_colors_determined = True
                print("Forced team colors with default values")
                return True
            return False
            
        # Create masks for green and white jerseys
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get player HSV values and count white vs green distribution
        white_players = 0
        green_players = 0
        
        for object_id, features in self.player_color_features.items():
            if not features or len(features) == 0:
                continue
            
            hsv_value = features[0]  # Use dominant color
            
            # Check if it's white (bright with low saturation)
            if hsv_value[1] < 30 and hsv_value[2] > 150:
                team_a_colors.append(hsv_value)
                white_players += 1
            # Check if it's green (hue in green range with good saturation)
            elif 45 <= hsv_value[0] <= 95 and hsv_value[1] > 70:
                team_b_colors.append(hsv_value)
                green_players += 1
                
        # Determine dominant colors
        if len(team_a_colors) >= 2 and len(team_b_colors) >= 2:
            # We have enough samples for both teams
            self.team_a_dominant_colors = team_a_colors  # White jerseys
            self.team_b_dominant_colors = team_b_colors  # Green jerseys
            self.team_colors_determined = True
            print(f"Team colors determined: Team A (White) has {len(team_a_colors)} samples, Team B (Green) has {len(team_b_colors)} samples")
            return True
            
        # Force default team colors if we couldn't determine them naturally
        self.team_a_dominant_colors = [[0, 0, 200]]  # White jersey in HSV
        self.team_b_dominant_colors = [[60, 180, 180]]  # Green jersey in HSV
        self.team_colors_determined = True
        print("Using default team colors after analysis")
        return True
    
    def determine_player_team(self, object_id):
        """Determine which team a player belongs to based on jersey color"""
        # If team colors haven't been determined, use fixed team colors
        if not self.team_colors_determined and hasattr(config, 'USE_FIXED_TEAM_COLORS') and config.USE_FIXED_TEAM_COLORS:
            self.team_a_dominant_colors = [config.TEAM_A_HSV_COLOR]
            self.team_b_dominant_colors = [config.TEAM_B_HSV_COLOR]
            self.team_colors_determined = True
            
        # Check if we already know the team for this player
        if object_id in self.teams:
            # Make sure player_teams is also updated for consistency
            if self.teams[object_id] == "team_a":
                self.player_teams[object_id] = "A"
            elif self.teams[object_id] == "team_b":
                self.player_teams[object_id] = "B"
            else:
                self.player_teams[object_id] = "Unknown"
            return self.teams[object_id]
            
        # First handle case where player color features are not available
        if object_id not in self.player_color_features:
            # Default to unknown team
            self.teams[object_id] = "unknown"
            self.player_teams[object_id] = "Unknown"
            return "unknown"
            
        # Get the dominant color feature
        dominant_color = self.player_color_features[object_id]
        
        # If no valid color was extracted
        if dominant_color is None:
            self.teams[object_id] = "unknown"
            self.player_teams[object_id] = "Unknown"
            return "unknown"
        
        # Direct check for white jersey (Team A)
        # White jersey: low saturation (S), high brightness (V)
        if dominant_color[1] < 30 and dominant_color[2] > 150:
            self.teams[object_id] = "team_a"
            self.player_teams[object_id] = "A"
            return "team_a"
            
        # Direct check for green jersey (Team B)
        # Green jersey: green hue range (H), higher saturation (S)
        if 45 <= dominant_color[0] <= 95 and dominant_color[1] > 70:
            self.teams[object_id] = "team_b"
            self.player_teams[object_id] = "B"
            return "team_b"
            
        # If team colors are determined, calculate similarity to each team
        if self.team_colors_determined and self.team_a_dominant_colors and self.team_b_dominant_colors:
            # Calculate distance to team A colors
            min_dist_a = float('inf')
            for color in self.team_a_dominant_colors:
                dist = np.sqrt(np.sum((np.array(dominant_color) - np.array(color))**2))
                if dist < min_dist_a:
                    min_dist_a = dist
                    
            # Calculate distance to team B colors
            min_dist_b = float('inf')
            for color in self.team_b_dominant_colors:
                dist = np.sqrt(np.sum((np.array(dominant_color) - np.array(color))**2))
                if dist < min_dist_b:
                    min_dist_b = dist
            
            # Determine team based on closest color distance with optional bias
            white_bias = 0.5  # Default even bias
            if hasattr(config, 'WHITE_TEAM_BIAS'):
                white_bias = config.WHITE_TEAM_BIAS
                
            if min_dist_a * white_bias < min_dist_b * (1 - white_bias):
                self.teams[object_id] = "team_a"
                self.player_teams[object_id] = "A"
            else:
                self.teams[object_id] = "team_b"
                self.player_teams[object_id] = "B"
        else:
            # If no clear determination can be made, default to unknown
            self.teams[object_id] = "unknown"
            self.player_teams[object_id] = "Unknown"
            
        return self.teams[object_id]
    
    def get_player_team(self, object_id):
        """Get the team of a player"""
        if object_id in self.teams:
            return self.teams[object_id]
        
        # If not already assigned, try to determine team
        if self.team_colors_determined and object_id in self.player_color_features:
            team = self.determine_player_team(object_id)
            return team
        
        # Force assignment based on object_id parity if enabled
        if hasattr(config, 'FORCE_TEAM_ASSIGNMENT') and config.FORCE_TEAM_ASSIGNMENT:
            # Assign based on ID parity for balanced distribution
            team_a_count = sum(1 for team in self.teams.values() if team == "team_a")
            team_b_count = sum(1 for team in self.teams.values() if team == "team_b")
            
            # Balance teams - assign to team with fewer players
            if team_a_count < team_b_count:
                return "team_a"
            elif team_b_count < team_a_count:
                return "team_b"
            else:
                # If balanced, use object ID
                return "team_a" if object_id % 2 == 0 else "team_b"
        
        return "unknown"
    
    def get_player_velocity(self, object_id):
        """Get the velocity of a player in meters per second"""
        return self.velocities.get(object_id, 0.0)
    
    def get_player_distance(self, object_id):
        """Get the total distance traveled by a player in meters"""
        return self.total_distances.get(object_id, 0.0)
    
    def get_player_tracking_time(self, object_id):
        """Get the time the player has been tracked in seconds"""
        if object_id in self.tracking_start_times:
            return time.time() - self.tracking_start_times[object_id]
        return 0.0
    
    def get_all_player_stats(self):
        """Get statistics for all tracked players"""
        stats = {}
        
        min_tracking_history = config.MIN_TRACKING_HISTORY if hasattr(config, 'MIN_TRACKING_HISTORY') else 10
        
        for object_id in list(self.objects.keys()):
            # Only include players with sufficient tracking history
            if (object_id in self.position_history and 
                len(self.position_history[object_id]) >= min_tracking_history):
                
                stats[object_id] = {
                    "team": self.get_player_team(object_id),
                    "velocity": self.get_player_velocity(object_id),
                    "distance": self.get_player_distance(object_id),
                    "tracking_time": self.get_player_tracking_time(object_id),
                    "centroid": self.objects[object_id]["centroid"],
                    "bbox": self.objects[object_id]["bbox"]
                }
        
        return stats 

    def balance_teams(self):
        """Ensure teams have a similar number of players by reassigning some if necessary"""
        # Count players in each team
        team_a_count = sum(1 for team in self.teams.values() if team == "team_a")
        team_b_count = sum(1 for team in self.teams.values() if team == "team_b")
        
        # If team counts are already balanced or close, no need to adjust
        if abs(team_a_count - team_b_count) <= 1:
            return
        
        # Determine which team has excess players
        if team_a_count > team_b_count + 1:
            # Too many in Team A, move some to Team B
            excess_team = "team_a"
            target_team = "team_b"
            excess_count = team_a_count - team_b_count
        else:
            # Too many in Team B, move some to Team A
            excess_team = "team_b"
            target_team = "team_a"
            excess_count = team_b_count - team_a_count
        
        # Adjust to get closer to balanced (move half the difference)
        to_move = excess_count // 2
        
        # Find candidates to move (those with ambiguous colors or least confident assignments)
        candidates = []
        for object_id, team in self.teams.items():
            if team == excess_team and object_id in self.objects:
                # Object is still being tracked
                if object_id in self.player_color_features:
                    color = self.player_color_features[object_id]
                    # Low saturation and mid value are more ambiguous colors
                    ambiguity = 1.0 if color is None else (
                        1.0 if 20 < color[1] < 70 and 100 < color[2] < 200 else 0.0
                    )
                    candidates.append((object_id, ambiguity))
        
        # Sort by ambiguity (most ambiguous first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Move the most ambiguous objects to the other team
        for i, (object_id, _) in enumerate(candidates):
            if i < to_move:
                self.teams[object_id] = target_team
            else:
                break
                
        # Update player_teams mapping to stay in sync with teams
        self.update_player_teams_map()

    def register_new_detections(self, detections, frame):
        """Register new detections and update existing objects"""
        if not detections or len(detections) == 0:
            return self.objects
            
        # If we have no objects currently being tracked, register all detections as new objects
        if len(self.objects) == 0:
            for detection in detections:
                # Extract the detection data
                if len(detection) == 6:
                    x, y, width, height, confidence, class_id = detection
                elif len(detection) == 5:
                    x, y, width, height, confidence = detection
                    class_id = 0  # Default to person class
                else:
                    continue  # Skip invalid format
                    
                centroid = (x, y)
                bbox = (int(x - width/2), int(y - height/2), width, height)
                self.register(centroid, bbox, frame)
            return self.objects
            
        # Calculate centroids for all current objects
        object_centroids = []
        object_ids = []
        for object_id, obj in self.objects.items():
            object_centroids.append(obj["centroid"])
            object_ids.append(object_id)
        
        # Calculate centroids for all new detections
        detection_centroids = []
        for detection in detections:
            x, y = detection[0:2]  # First two elements are x,y coordinates
            detection_centroids.append((x, y))
            
        # Calculate distances between existing centroids and new detections
        D = dist.cdist(np.array(object_centroids), np.array(detection_centroids))
        
        # Use Hungarian algorithm to find optimal matching
        row_ind, col_ind = linear_sum_assignment(D)
        
        # Track used objects and detections
        used_objects = set()
        used_detections = set()
        
        # Go through the matches
        for (row, col) in zip(row_ind, col_ind):
            # If the distance is less than our maximum distance threshold
            if D[row, col] < self.max_distance:
                # Get the ID of the matched object
                object_id = object_ids[row]
                
                # Get the detection
                detection = detections[col]
                
                # Add to lists of used rows and columns
                used_objects.add(object_id)
                used_detections.add(col)
                
                # Extract coordinates
                if len(detection) == 6:
                    x, y, width, height, confidence, class_id = detection
                elif len(detection) == 5:
                    x, y, width, height, confidence = detection
                    class_id = 0  # Default to person class
                else:
                    continue  # Skip invalid format
                    
                centroid = (x, y)
                bbox = (int(x - width/2), int(y - height/2), width, height)
                
                # Update the object
                self.objects[object_id]["centroid"] = centroid
                self.objects[object_id]["bbox"] = bbox
                self.objects[object_id]["disappeared"] = 0
                
                # Update trajectory
                if object_id in self.trajectories:
                    self.trajectories[object_id].append(centroid)
                else:
                    self.trajectories[object_id] = [centroid]
                    
                # Update velocity and distance
                self.update_velocity_and_distance(object_id)
                
                # Update color features
                self.extract_color_features(object_id, frame)
                
        # Check for disappeared objects
        for object_id in list(self.objects.keys()):
            if object_id not in used_objects:
                # Increment disappeared counter
                self.objects[object_id]["disappeared"] += 1
                
                # Deregister if it's been gone for too long
                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    self.deregister(object_id)
                    
        # Register each new detection as a new object
        for i, detection in enumerate(detections):
            if i not in used_detections:
                # Extract coordinates
                if len(detection) == 6:
                    x, y, width, height, confidence, class_id = detection
                elif len(detection) == 5:
                    x, y, width, height, confidence = detection
                    class_id = 0  # Default to person class
                else:
                    continue  # Skip invalid format
                    
                centroid = (x, y)
                bbox = (int(x - width/2), int(y - height/2), width, height)
                self.register(centroid, bbox, frame)
                
        return self.objects
        
    def register_object(self, detection, objects_dict, frame):
        """Register a new object with the next available ID from a detection"""
        try:
            # Extract coordinates - handle both list format and dictionary format
            if isinstance(detection, list) or isinstance(detection, tuple):
                if len(detection) >= 5:
                    x, y, width, height, confidence = detection[:5]
                    class_id = detection[5] if len(detection) > 5 else 0
                else:
                    return  # Invalid detection format
            else:
                # Dictionary format
                if "centroid" in detection and "bbox" in detection:
                    x, y = detection["centroid"]
                    bbox = detection["bbox"]
                    width, height = bbox[2], bbox[3]
                    confidence = detection.get("confidence", 1.0)
                    class_id = detection.get("class_id", 0)
                else:
                    return  # Invalid detection format
            
            # Calculate centroid and rect
            centroid = (float(x), float(y))
            rect = (int(float(x) - float(width)/2), int(float(y) - float(height)/2), 
                    float(width), float(height))
            
            # Register the new object
            objects_dict[self.next_object_id] = {
                "centroid": centroid,
                "centroids": [centroid],
                "rect": rect,
                "bbox": rect,  # Add bbox for compatibility
                "disappeared": 0,
                "confidence": confidence,
                "class": class_id,
                "timestamps": [time.time()],
                "positions": [centroid],
                "velocity": 0,
                "distance": 0,
                "team": None
            }
            
            # Initialize trajectory
            self.trajectories[self.next_object_id] = [centroid]
            
            # Initialize position history
            self.position_history[self.next_object_id] = [centroid]
            
            # Initialize frames tracked
            self.frames_tracked[self.next_object_id] = 1
            
            # Initialize tracking time
            self.tracking_start_times[self.next_object_id] = time.time()
            
            # Initialize velocities and distances
            self.velocities[self.next_object_id] = 0.0
            self.total_distances[self.next_object_id] = 0.0
            
            # Extract color features
            if frame is not None:
                self.extract_color_features(self.next_object_id, frame)
            
            # Determine team if possible
            if self.team_colors_determined:
                team = self.determine_player_team(self.next_object_id)
                self.teams[self.next_object_id] = team
                
                # Update player_teams mapping
                if team == "team_a":
                    self.player_teams[self.next_object_id] = "A"
                elif team == "team_b":
                    self.player_teams[self.next_object_id] = "B"
                else:
                    self.player_teams[self.next_object_id] = "Unknown"
            else:
                self.teams[self.next_object_id] = "unknown"
                self.player_teams[self.next_object_id] = "Unknown"
            
            # Increment next object ID
            self.next_object_id += 1
        except Exception as e:
            print(f"Error in register_object: {e}")
            print(f"Detection data: {detection}")

    def collect_team_colors(self):
        """Collect color samples from tracked players for team classification"""
        if self.team_colors_determined:
            return
            
        for object_id, obj in self.objects.items():
            if object_id in self.player_color_features:
                # Only add if not already in list
                if object_id not in self.player_ids_for_colors:
                    color = self.player_color_features[object_id]
                    self.color_samples.append(color)
                    self.player_ids_for_colors.append(object_id)
                
        # Try to determine team colors if we have enough samples
        if (not self.team_colors_determined and 
            len(self.color_samples) >= self.min_players_for_classification and
            self.team_class_attempts < 3):  # Limit attempts
            
            self.determine_team_colors(self.last_frame)
            self.team_class_attempts += 1 

    def bbox_iou(self, box1, box2):
        """Calculate IoU (Intersection over Union) for two bounding boxes"""
        try:
            # Extract box coordinates
            # box1 and box2 are in (x, y, w, h) format
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Convert to (x1, y1, x2, y2) format
            x1_1, y1_1 = x1, y1
            x2_1, y2_1 = x1 + w1, y1 + h1
            x1_2, y1_2 = x2, y2
            x2_2, y2_2 = x2 + w2, y2 + h2
            
            # Calculate area of each box
            area1 = w1 * h1
            area2 = w2 * h2
            
            # Calculate coordinates of intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            # Calculate intersection width and height
            inter_width = max(0, xi2 - xi1)
            inter_height = max(0, yi2 - yi1)
            
            # Calculate intersection area
            inter_area = inter_width * inter_height
            
            # Calculate union area
            union_area = area1 + area2 - inter_area
            
            # Calculate IoU
            iou = inter_area / max(union_area, 1e-6)
            return iou
        except Exception as e:
            print(f"Error in bbox_iou: {e}")
            return 0.0 

    def calculate_velocity(self, object_id):
        """Calculate velocity for an object based on its trajectory history"""
        try:
            # Check if we have enough trajectory points
            if object_id not in self.trajectories or len(self.trajectories[object_id]) < 2:
                return 0.0
                
            # Get the last two points
            current_pos = np.array(self.trajectories[object_id][-1])
            prev_pos = np.array(self.trajectories[object_id][-2])
            
            # Calculate distance moved in pixels
            distance_pixels = np.linalg.norm(current_pos - prev_pos)
            
            # Convert pixels to meters
            pixels_to_meters = config.PIXELS_TO_METERS if hasattr(config, 'PIXELS_TO_METERS') else 0.1
            distance_meters = distance_pixels * pixels_to_meters
            
            # Calculate time between frames
            fps = config.FPS if hasattr(config, 'FPS') else 30.0
            time_seconds = 1.0 / fps
            
            # Calculate velocity in meters per second
            velocity_mps = distance_meters / time_seconds
            
            # Convert to km/h
            velocity_kmh = velocity_mps * 3.6
            
            # Apply maximum realistic speed filter
            max_speed = config.MAX_REALISTIC_SPEED if hasattr(config, 'MAX_REALISTIC_SPEED') else 40.0  # km/h
            if velocity_kmh > max_speed:
                velocity_kmh = self.velocities.get(object_id, 0.0)  # Keep previous value if unrealistic
                
            # Store velocity
            self.velocities[object_id] = velocity_kmh
            
            return velocity_kmh
        except Exception as e:
            print(f"Error calculating velocity for object {object_id}: {e}")
            return 0.0 