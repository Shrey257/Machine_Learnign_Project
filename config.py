# Configuration parameters for football analysis optimized for football match with white and green teams

import os
import numpy as np
import cv2

# Paths
INPUT_VIDEO_PATH = os.path.join("input", "Raw_Data.mp4")
OUTPUT_VIDEO_PATH = os.path.join("output", "videos", "final_output.mp4")
MODEL_WEIGHTS = os.path.join("models", "football-players-detection", "weights", "best.pt")

# Detection parameters - optimized for player detection
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to detect more players
IOU_THRESHOLD = 0.25  # Lower for better overlap handling
NMS_THRESHOLD = 0.30  # Adjusted for better detection
DETECTION_INTERVAL = 1  # Process every frame for better tracking

# Multiple detection scales to catch all players
DETECTION_SIZES = [640, 960]  # Two sizes for better coverage

# Player tracking parameters
MAX_DISAPPEARED = 15  # Longer tracking window to maintain player IDs
MAX_DISTANCE = 80  # Larger distance for tracking fast-moving players

# Team classification parameters
TEAM_COLORS = {
    "team_a": [255, 255, 255],  # White jerseys (BGR)
    "team_b": [0, 200, 0]       # Green jerseys (BGR)
}

# Velocity calculation parameters
VELOCITY_WINDOW = 5  # Frames to calculate average velocity
PIXELS_TO_METERS = 0.25  # Conversion ratio from pixels to meters (calibrated for football field)
FPS = 30.0  # Default FPS value
MAX_REALISTIC_SPEED = 36.0  # Maximum realistic speed in km/h for a football player

# Data quality filters
MAX_REALISTIC_DISTANCE = 5000.0  # Maximum realistic distance in m for entire match
FILTER_UNREALISTIC_VALUES = True  # Filter out unrealistic values in reports

# HTML Report configuration
HTML_REPORT_PATH = os.path.join("output", "reports", "player_stats_report.html")
SAVE_PLAYER_STATS = True
GENERATE_REPORT = True

# CLI added parameters (default values)
VERBOSE = True  # Enable verbose output for debugging
DISPLAY_VIDEO = False
SKIP_FRAMES = 0  # Don't skip frames to ensure all players are detected
DISABLE_TEAM_DETECTION = False

# Visualization colors - football team colors
TEAM_A_COLOR = (255, 255, 255)  # White for team A (BGR format) - white jerseys
TEAM_B_COLOR = (0, 200, 0)      # Green for team B (BGR format) - green jerseys
UNKNOWN_TEAM_COLOR = (128, 128, 128)  # Gray for unknown team
BALL_COLOR = (0, 215, 255)  # Yellow for ball

# Text visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7  # Larger font for better visibility
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White text
TEXT_BG_COLOR = (0, 0, 0)  # Black background
LINE_THICKNESS = 2

# Field detection parameters - specific to football field in image
FIELD_COLOR_LOWER = (35, 40, 40)  # HSV lower bounds for green field
FIELD_COLOR_UPPER = (90, 255, 255)  # HSV upper bounds for field color

# Player filtering parameters - football player proportions
MIN_PLAYER_ASPECT_RATIO = 0.15  # More permissive for distant players
MAX_PLAYER_ASPECT_RATIO = 0.85  # Adjusted for players with arms out
MIN_TRACKING_HISTORY = 2  # Lower to see players sooner
MIN_PLAYER_SIZE = 200  # Smaller minimum area to detect distant players

# Appearance matching parameters
APPEARANCE_WEIGHT = 0.7  # Higher weight for appearance in team matching
COLOR_SIMILARITY_THRESHOLD = 0.3  # Threshold for team assignment

# Team classification improvements - based on image shown
MIN_PLAYERS_FOR_TEAM_CLASSIFICATION = 2  # Start classifying with fewer players
COLOR_CLUSTER_K = 2  # Number of color clusters to extract
JERSEY_ROI_RATIO = 0.4  # Focus on upper body for jersey color

# Additional filtering parameters
EDGE_MARGIN = 10  # Margin to detect players near edges
FORCE_TEAM_ASSIGNMENT = True  # Force unknown players to be assigned to a team
ENFORCE_BALANCED_TEAMS = True  # Try to maintain similar number of players in each team
MIN_TEAM_SIZE = 5  # Minimum players per team to enforce balanced assignment
TEAM_BALANCE_INTERVAL = 10  # How often to check and rebalance teams

# Parameters specifically for the football match in image
WHITE_JERSEY_HSV_RANGE_LOW = [0, 0, 150]  # White/light jerseys (Team A) in HSV
WHITE_JERSEY_HSV_RANGE_HIGH = [180, 30, 255]
GREEN_JERSEY_HSV_RANGE_LOW = [45, 50, 50]  # Green jerseys (Team B) in HSV
GREEN_JERSEY_HSV_RANGE_HIGH = [95, 255, 200]

# Fixed team colors based on the image
USE_FIXED_TEAM_COLORS = True  # Use pre-defined team colors from the image
TEAM_A_HSV_COLOR = [0, 0, 200]  # White jerseys in HSV
TEAM_B_HSV_COLOR = [60, 150, 150]  # Green jerseys in HSV
WHITE_TEAM_BIAS = 0.6  # Bias team assignment toward white jerseys for Team A
