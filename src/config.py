import torch
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    QUERY_DIR = os.path.join(DATA_DIR, "query")
    SKU_DIR = os.path.join(DATA_DIR, "sku")
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Matching Weights
    COLOR_WEIGHT = 0.25
    FEATURE_WEIGHT = 0.75
    
    # Feature Extraction
    MAX_KEYPOINTS = 1024
    RESIZE_MAX_EDGE = 640
    
    # Color Histogram
    HIST_BINS = [50, 60]
    HIST_RANGES = [0, 180, 0, 256]
    
    # Processing
    UPSCALE_FACTOR = 2.5  # For ultra quality extraction
    
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.QUERY_DIR, exist_ok=True)
        os.makedirs(Config.SKU_DIR, exist_ok=True)

# Print configuration on load
print(f"Configuration loaded. Device: {Config.DEVICE}")
print(f"Base Directory: {Config.BASE_DIR}")
