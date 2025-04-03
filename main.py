import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
from ultralytics import YOLO
import torch
from player_tracker import PlayerTracker
import config
from report_generator import ReportGenerator
import argparse
from tqdm import tqdm

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def calculate_centroid(bbox):
    """Calculate centroid from bounding box"""
    x, y, w, h = bbox
    return (int(x + w/2), int(y + h/2))

def create_field_mask(frame, field_color_lower=(35, 40, 40), field_color_upper=(90, 255, 255)):
    """Create a mask for the playing field based on color"""
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask based on the field color (green)
    field_mask = cv2.inRange(hsv, field_color_lower, field_color_upper)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((15, 15), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate to include players standing on the field edges
    field_mask = cv2.dilate(field_mask, np.ones((35, 35), np.uint8))
    
    # Find the largest contour (should be the field)
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a clean mask with only the largest contour
        clean_mask = np.zeros_like(field_mask)
        cv2.drawContours(clean_mask, [largest_contour], 0, 255, -1)
        
        # Apply closing operation to fill any small holes
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        
        return clean_mask
    
    return field_mask

def is_valid_player_aspect_ratio(bbox, min_aspect=0.2, max_aspect=0.7):
    """Check if the bounding box has a reasonable aspect ratio for a person"""
    x, y, w, h = bbox
    
    # Skip if dimensions are invalid
    if w <= 0 or h <= 0:
        return False
    
    # Calculate aspect ratio (width / height)
    aspect_ratio = w / h
    
    # Check if aspect ratio is within expected range for a standing person
    return min_aspect <= aspect_ratio <= max_aspect

def enhance_detections(detections, field_mask=None, confidence_threshold=0.40):
    """Filter detections based on confidence, field position, and aspect ratio"""
    enhanced_detections = []
    
    for det in detections:
        # Extract info from detection
        bbox = det["bbox"]
        conf = det["confidence"]
        cls = det["class"]
        
        # Skip if confidence is too low
        if conf < confidence_threshold:
            continue
        
        # Extract bounding box dimensions
        x, y, w, h = bbox
        
        # Filter out detections that are too small
        min_player_size = config.MIN_PLAYER_SIZE if hasattr(config, 'MIN_PLAYER_SIZE') else 1000
        if w * h < min_player_size:
            continue
            
        # Filter out detections at the edges of the frame
        edge_margin = config.EDGE_MARGIN if hasattr(config, 'EDGE_MARGIN') else 20
        if x < edge_margin or y < edge_margin:
            continue
        
        # Skip if aspect ratio is not valid for a player
        if not is_valid_player_aspect_ratio(bbox, 
                                           config.MIN_PLAYER_ASPECT_RATIO if hasattr(config, 'MIN_PLAYER_ASPECT_RATIO') else 0.15,
                                           config.MAX_PLAYER_ASPECT_RATIO if hasattr(config, 'MAX_PLAYER_ASPECT_RATIO') else 0.8):
            continue
        
        # Check if detection is on the field using the mask
        if field_mask is not None:
            # Use the bottom center of the bounding box (player's feet)
            foot_x = int(x + w/2)
            foot_y = int(y + h)
            
            # Ensure coordinates are within image bounds
            h_img, w_img = field_mask.shape[:2]
            if 0 <= foot_x < w_img and 0 <= foot_y < h_img:
                # Check if this point is on the field
                # Make the check more strict - player must be on the field
                is_on_field = False
                
                # Direct check
                if field_mask[foot_y, foot_x] > 0:
                    is_on_field = True
                else:
                    # Check points around the feet in a small radius
                    radius = 10  # reduced radius for more strict checking
                    for ry in range(max(0, foot_y-radius), min(h_img, foot_y+radius)):
                        for rx in range(max(0, foot_x-radius), min(w_img, foot_x+radius)):
                            if field_mask[ry, rx] > 0:
                                is_on_field = True
                                break
                        if is_on_field:
                            break
                
                if not is_on_field:
                    continue
        
        enhanced_detections.append(det)
    
    return enhanced_detections

def run_multi_scale_detection(model, frame):
    """
    Run multi-scale detection on a frame to find players and the ball
    Uses multiple scales for better detection of all players
    """
    # Store player and ball detections across all scales
    all_player_detections = []
    ball_detections = []
    
    # Use multiple scales for better detection
    detection_sizes = config.DETECTION_SIZES
    
    # Track detection performance
    start_time = time.time()
    
    try:
        # Enhance the frame for better detection
        enhanced_frame = enhance_frame_for_detection(frame)
        
        # Run detection at each scale
        for detection_size in detection_sizes:
            results = model(enhanced_frame, imgsz=detection_size)
            
            # Process results
            if results and len(results) > 0:
                for result in results:
                    if hasattr(result, 'boxes'):
                        # For each detected box
                        for box in result.boxes.data.cpu().numpy():
                            x1, y1, x2, y2, confidence, class_id = box
                            
                            # Skip low confidence detections
                            if confidence < config.CONFIDENCE_THRESHOLD:
                                continue
                            
                            # Calculate width and height
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Calculate center
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            # Format: (center_x, center_y, width, height, confidence, class_id)
                            detection = [center_x, center_y, width, height, confidence, class_id]
                            
                            # Class 0 is person and class 32 is sports ball in COCO
                            if int(class_id) == 0:  # Person
                                if width * height > config.MIN_PLAYER_SIZE:  # Filter out too small detections
                                    all_player_detections.append(detection)
                            elif int(class_id) == 32:  # Sports ball
                                ball_detections.append(detection)
        
        # For the ball, take the highest confidence detection
        ball_detection = None
        if ball_detections:
            # Sort by confidence (descending)
            ball_detections.sort(key=lambda x: x[4], reverse=True)
            ball_detection = ball_detections[0]
        
        # Remove duplicate player detections through non-max suppression
        final_player_detections = []
        if all_player_detections:
            # Convert to expected format for NMS
            boxes = []
            for det in all_player_detections:
                x, y, w, h, conf, cls = det
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2
                boxes.append([x1, y1, x2, y2, conf, cls])
                
            # Apply NMS
            boxes = np.array(boxes)
            scores = boxes[:, 4]
            indices = cv2.dnn.NMSBoxes(
                boxes[:, :4].tolist(),
                scores.tolist(),
                config.CONFIDENCE_THRESHOLD,
                config.IOU_THRESHOLD
            ).flatten()
            
            # Convert back to center format
            for i in indices:
                x1, y1, x2, y2, conf, cls = boxes[i]
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                final_player_detections.append([x, y, w, h, conf, cls])
        
        if config.VERBOSE and len(final_player_detections) > 0:
            print(f"Detected {len(final_player_detections)} players and {1 if ball_detection is not None else 0} ball")
            print(f"Detection time: {time.time() - start_time:.3f}s")
            
        return final_player_detections, ball_detection
        
    except Exception as e:
        print(f"Error in detection: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def non_max_suppression(detections):
    """Custom non-maximum suppression for player detections"""
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    # Initialize list of picked indexes
    picked = []
    
    # Initialize bounding boxes and scores
    boxes = np.array([d["bbox"] for d in detections])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = np.array([d["confidence"] for d in detections])
    
    # Compute area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Process boxes in order of confidence
    indices = np.arange(len(scores))
    
    while len(indices) > 0:
        # Pick the box with the highest score
        last = len(indices) - 1
        i = indices[0]
        picked.append(i)
        
        # Find coordinates of intersection
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        # Compute intersection area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        
        # Compute IoU
        union = area[i] + area[indices[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        indices = np.delete(indices, np.concatenate(([0], np.where(iou > config.NMS_THRESHOLD)[0] + 1)))
    
    return [detections[i] for i in picked]

def enhance_frame_for_detection(frame):
    """Enhance the frame for better detection performance"""
    # Apply contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame

def run_analysis(input_video, output_video, weights, config_path=None):
    """
    Run the football analysis on the input video
    
    Args:
        input_video: Path to the input video
        output_video: Path to save the output video
        weights: Path to the model weights
        config_path: Path to the configuration file
    """
    # Load the video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Load detection model
    model = YOLO(weights)
    print(f"Loaded model from {weights}")
    
    # Initialize trackers (just one tracker for both players and ball)
    player_tracker = PlayerTracker()
    
    # Create the report generator
    report_gen = None
    if hasattr(config, 'GENERATE_REPORT') and config.GENERATE_REPORT:
        os.makedirs(os.path.dirname(config.HTML_REPORT_PATH), exist_ok=True)
        report_gen = ReportGenerator(config.HTML_REPORT_PATH)
    
    # Process video frame by frame
    frame_count = 0
    
    # For faster processing, skip frames
    frame_skip = 3  # Process every 4th frame for maximum speed
    
    # Process frames with progress bar
    print(f"Processing video with {total_frames} frames...")
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    # For tracking detection performance
    detection_times = []
    last_save_frame = 0
    
    # Detection interval counter
    detection_counter = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only run detection every few frames for speed
            run_detection = detection_counter % (frame_skip + 1) == 0
            
            if run_detection:
                # Track detection time
                detection_start = time.time()
                
                # Run detection
                player_detections, ball_detection = run_multi_scale_detection(model, frame)
                
                # Register new detections
                register_new_detections(player_tracker, player_detections, ball_detection, frame)
                
                # Calculate detection time
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                
                # Print average detection time every 30 detections
                if len(detection_times) % 30 == 0:
                    avg_time = sum(detection_times[-30:]) / 30
                    print(f"Average detection time: {avg_time:.2f}s per frame")
            
            # Update metrics for all tracked players
            for object_id in list(player_tracker.objects.keys()):
                # Skip ball (id 0)
                if object_id == 0:
                    continue
                player_tracker.update_velocity_and_distance(object_id)
            
            # Collect color samples for team classification
            player_tracker.collect_team_colors()
            
            # Balance teams if needed (less frequently for speed)
            if hasattr(config, 'TEAM_BALANCE_INTERVAL') and frame_count % config.TEAM_BALANCE_INTERVAL == 0:
                balance_teams(player_tracker)
            
            # Draw tracked objects on the frame
            processed_frame = draw_tracked_objects(frame, player_tracker)
            
            # Write the frame to the output video
            out.write(processed_frame)
            
            # Update report data
            if report_gen:
                report_gen.update_frame_data(
                    frame_count, 
                    player_tracker.player_teams, 
                    player_tracker.velocities, 
                    player_tracker.total_distances
                )
            
            # Periodically save report (every ~300 frames)
            if report_gen and frame_count - last_save_frame >= 300:
                try:
                    # Generate interim report
                    report_gen.generate_report()
                    print(f"Intermediate report saved at frame {frame_count}")
                    last_save_frame = frame_count
                except Exception as e:
                    print(f"Error saving intermediate report: {e}")
            
            # Update progress bar
            pbar.update(1)
            frame_count += 1
            detection_counter += 1
    
    except KeyboardInterrupt:
        print("Processing interrupted! Saving progress...")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close progress bar
        pbar.close()
        
        # Release resources
        cap.release()
        out.release()
        
        # Always generate final report
        if report_gen:
            try:
                report_gen.generate_report()
                print(f"Report generated at {config.HTML_REPORT_PATH}")
            except Exception as e:
                print(f"Error generating final report: {e}")
        
        print(f"Analysis complete. Output saved to {output_video}")
        return player_tracker

def balance_teams(player_tracker):
    """Ensure balanced team assignment between Team A and Team B"""
    # Count players in each team
    teams = player_tracker.teams
    team_a_count = sum(1 for team in teams.values() if team == "team_a")
    team_b_count = sum(1 for team in teams.values() if team == "team_b")
    
    print(f"Initial team counts: Team A: {team_a_count}, Team B: {team_b_count}")
    
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
    
    # Find candidates to move (those with ambiguous colors)
    candidates = []
    for object_id, team in teams.items():
        if team == excess_team and object_id in player_tracker.objects:
            # Calculate ambiguity score based on color
            ambiguity = 0.0
            if object_id in player_tracker.color_features:
                color = player_tracker.color_features[object_id]
                # Low saturation and mid value are more ambiguous
                if color is not None:
                    # More ambiguous if not clearly white or green
                    if not (color[1] < 30 and color[2] > 150) and not (45 <= color[0] <= 95 and color[1] > 70):
                        ambiguity = 1.0
            candidates.append((object_id, ambiguity))
    
    # Sort by ambiguity (most ambiguous first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Move the most ambiguous objects to the other team
    for i, (object_id, _) in enumerate(candidates):
        if i < to_move:
            player_tracker.teams[object_id] = target_team
        else:
            break
    
    # Count again after balancing
    team_a_count = sum(1 for team in teams.values() if team == "team_a")
    team_b_count = sum(1 for team in teams.values() if team == "team_b")
    print(f"Balanced team counts: Team A: {team_a_count}, Team B: {team_b_count}")

def draw_tracked_objects(frame, player_tracker, min_tracking_history=2):
    """
    Draw tracked objects on the frame with player IDs, velocity, and distance information
    Ensures all information is clearly visible
    """
    # Make a copy of the frame to avoid modifying the original
    output_frame = frame.copy()
    
    # Pre-calculate font settings for text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.FONT_SCALE
    thickness = config.FONT_THICKNESS
    
    # Draw tracked players
    for object_id, obj in player_tracker.objects.items():
        # Skip ball (ID 0) - handle it separately
        if object_id == 0:
            continue
            
        # Only draw players that have been tracked for some time
        if object_id not in player_tracker.position_history or len(player_tracker.position_history.get(object_id, [])) < min_tracking_history:
            continue
            
        # Get the current position and bounding box
        if "rect" in obj:
            x, y, w, h = obj["rect"]
        elif "bbox" in obj:
            x, y, w, h = obj["bbox"]
        else:
            # If no bounding box info, try to use centroid
            if "centroid" in obj:
                cx, cy = obj["centroid"]
                # Create a small default box around the centroid
                w, h = 30, 60  # Default player size
                x, y = cx - w//2, cy - h//2
            else:
                continue  # Skip if no position info available
            
        # Ensure coordinates are integers
        x, y, w, h = int(x), int(y), int(w), int(h)
            
        # Determine the team color (red for team A, blue for team B)
        if player_tracker.player_teams.get(object_id) == "A":
            # Team A - Red (BGR)
            color = (0, 0, 255)  # Red team (Team A - white jerseys)
        elif player_tracker.player_teams.get(object_id) == "B":
            # Team B - Blue (BGR)
            color = (255, 0, 0)  # Blue team (Team B - green jerseys)
        else:
            # Unknown team - Gray
            color = (128, 128, 128)
            
        # Draw the bounding box
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
        
        # Get velocity and distance
        velocity = player_tracker.velocities.get(object_id, 0)
        distance = player_tracker.total_distances.get(object_id, 0)
        
        # Format text with player ID, velocity, and distance
        id_text = f"ID:{object_id}"
        vel_text = f"{velocity:.1f} km/h"
        dist_text = f"{distance:.1f}m"
        
        # Draw ID at the top of the bounding box with background
        id_text_size = cv2.getTextSize(id_text, font, font_scale, thickness)[0]
        cv2.rectangle(output_frame, 
                     (x, y - id_text_size[1] - 10), 
                     (x + id_text_size[0] + 10, y), 
                     color, -1)
        cv2.putText(output_frame, id_text, 
                   (x + 5, y - 5), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw velocity at the bottom of the bounding box
        vel_text_size = cv2.getTextSize(vel_text, font, font_scale, thickness)[0]
        cv2.rectangle(output_frame, 
                     (x, y + h), 
                     (x + vel_text_size[0] + 10, y + h + vel_text_size[1] + 10), 
                     color, -1)
        cv2.putText(output_frame, vel_text, 
                   (x + 5, y + h + vel_text_size[1] + 5), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw distance below velocity
        dist_text_size = cv2.getTextSize(dist_text, font, font_scale, thickness)[0]
        cv2.rectangle(output_frame, 
                     (x, y + h + vel_text_size[1] + 15), 
                     (x + dist_text_size[0] + 10, y + h + vel_text_size[1] + dist_text_size[1] + 25), 
                     color, -1)
        cv2.putText(output_frame, dist_text, 
                   (x + 5, y + h + vel_text_size[1] + dist_text_size[1] + 20), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw trajectory
        if object_id in player_tracker.position_history:
            history = player_tracker.position_history[object_id]
            # Use up to 20 positions for trajectory
            trajectory_points = history[-20:]
            
            # Draw trajectory lines if we have enough points
            if len(trajectory_points) > 1:
                # Convert to numpy array of integers for polylines
                points = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output_frame, [points], False, color, 2)
    
    # Draw the ball (ID 0) if it's being tracked
    ball_id = 0
    if ball_id in player_tracker.objects:
        ball = player_tracker.objects[ball_id]
        
        # Get ball position and bounding box
        if "rect" in ball:
            x, y, w, h = ball["rect"]
            center_x = x + w // 2
            center_y = y + h // 2
        elif "bbox" in ball:
            x, y, w, h = ball["bbox"]
            center_x = x + w // 2
            center_y = y + h // 2
        elif "centroid" in ball:
            center_x, center_y = ball["centroid"]
            w, h = 20, 20  # Default ball size
        else:
            return output_frame  # Skip if no position info
            
        # Ensure coordinates are integers
        center_x, center_y = int(center_x), int(center_y)
            
        # Draw the ball as a yellow circle with label
        ball_color = (0, 215, 255)  # Yellow in BGR
        cv2.circle(output_frame, (center_x, center_y), max(8, w // 2), ball_color, -1)
        
        # Label the ball
        ball_text = "BALL"
        text_size = cv2.getTextSize(ball_text, font, font_scale, thickness)[0]
        cv2.rectangle(output_frame, 
                    (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 25), 
                    (center_x + text_size[0]//2 + 5, center_y - 5), 
                    ball_color, -1)
        cv2.putText(output_frame, ball_text, 
                   (center_x - text_size[0]//2, center_y - 10), 
                   font, font_scale, (0, 0, 0), thickness)
        
        # Draw ball trajectory
        if ball_id in player_tracker.position_history:
            ball_history = player_tracker.position_history[ball_id]
            # Use up to 30 points for ball trajectory
            if len(ball_history) > 1:
                # Convert to numpy array for polylines
                ball_points = np.array(ball_history[-30:], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output_frame, [ball_points], False, ball_color, 3)
    
    return output_frame

def register_new_detections(player_tracker, player_detections, ball_detection, frame):
    """
    Register new detections with the appropriate tracker
    
    Args:
        player_tracker: The player tracker object
        player_detections: List of player detections
        ball_detection: Ball detection data or None
        frame: Current video frame
    """
    try:
        # Register player detections
        if player_detections:
            for detection in player_detections:
                # Check if detection is in the right format
                if len(detection) >= 4:  # At least need x, y, width, height
                    # Extract detection data
                    if isinstance(detection, (list, tuple, np.ndarray)):
                        # Format: [x, y, w, h, confidence, class_id]
                        x, y, w, h = detection[:4]
                        if len(detection) > 4:
                            confidence = float(detection[4])
                        else:
                            confidence = 0.5
                            
                        if len(detection) > 5:
                            class_id = int(detection[5])
                        else:
                            class_id = 0
                        
                        # Register directly with player tracker
                        try:
                            player_tracker.register(detection)
                        except Exception as e:
                            print(f"Error registering player detection: {e}")
                    else:
                        print(f"Unsupported detection format: {type(detection)}")
                else:
                    print(f"Detection doesn't have enough elements: {detection}")
        
        # Register ball detection with ID 0
        if ball_detection is not None:
            try:
                # Extract coordinates
                x, y, w, h = ball_detection[:4]
                confidence = ball_detection[4] if len(ball_detection) > 4 else 0.5
                class_id = 32  # YOLO class ID for sports ball
                
                # Explicitly create the ball detection with ID 0
                ball_center = (x, y)
                ball_rect = (int(x - w/2), int(y - h/2), int(w), int(h))
                
                # Check if ball already exists in tracker
                if 0 in player_tracker.objects:
                    # Update the existing ball tracking
                    player_tracker.objects[0]["centroid"] = ball_center
                    player_tracker.objects[0]["rect"] = ball_rect
                    player_tracker.objects[0]["bbox"] = ball_rect
                    player_tracker.objects[0]["disappeared"] = 0
                    
                    # Update ball position history
                    if 0 in player_tracker.position_history:
                        player_tracker.position_history[0].append(ball_center)
                    else:
                        player_tracker.position_history[0] = [ball_center]
                        
                    # Update ball trajectory
                    if 0 in player_tracker.trajectories:
                        player_tracker.trajectories[0].append(ball_center)
                    else:
                        player_tracker.trajectories[0] = [ball_center]
                else:
                    # First time seeing the ball, register it
                    # Force ID 0 for the ball
                    ball_id = 0
                    player_tracker.objects[ball_id] = {
                        "centroid": ball_center,
                        "centroids": [ball_center],
                        "rect": ball_rect,
                        "bbox": ball_rect,
                        "disappeared": 0,
                        "confidence": confidence,
                        "class": class_id,
                        "timestamps": [time.time()],
                        "velocity": 0,
                        "distance": 0,
                        "team": None
                    }
                    
                    # Initialize position history and trajectory
                    player_tracker.position_history[ball_id] = [ball_center]
                    player_tracker.trajectories[ball_id] = [ball_center]
                    player_tracker.velocities[ball_id] = 0.0
                    player_tracker.total_distances[ball_id] = 0.0
            except Exception as e:
                print(f"Error processing ball detection: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error in register_new_detections: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--input', type=str, help='Path to input video', default=config.INPUT_VIDEO_PATH)
    parser.add_argument('--output', type=str, help='Path to output video', default=config.OUTPUT_VIDEO_PATH)
    parser.add_argument('--weights', type=str, help='Path to YOLOv8 weights', default=config.MODEL_WEIGHTS)
    parser.add_argument('--config', type=str, help='Path to config file', default=None)
    
    args = parser.parse_args()
    
    # Run the analysis
    print(f"Starting analysis with:")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Weights: {args.weights}")
    
    # Run the analysis and get the tracker objects
    player_tracker = run_analysis(
        input_video=args.input,
        output_video=args.output,
        weights=args.weights,
        config_path=args.config
    )
    
    print(f"Analysis complete. Output video saved to {args.output}")
    
    if config.GENERATE_REPORT:
        print(f"HTML report generated at {config.HTML_REPORT_PATH}")

if __name__ == "__main__":
    main() 