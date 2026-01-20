import cv2
import numpy as np
import torch
from .config import Config

def load_image(path):
    return cv2.imread(path)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calculate_optimal_size(pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    return maxWidth, maxHeight

def extract_region_ultra_quality(img, pts):
    rect = order_points(pts)
    maxWidth, maxHeight = calculate_optimal_size(pts)
    
    upscale_factor = Config.UPSCALE_FACTOR
    dst_width = int(maxWidth * upscale_factor)
    dst_height = int(maxHeight * upscale_factor)
    
    dst = np.array([
        [0, 0],
        [dst_width - 1, 0],
        [dst_width - 1, dst_height - 1],
        [0, dst_height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    
    warped = cv2.warpPerspective(
        img, M, (dst_width, dst_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped

def preprocess_tensor(image, max_size=None):
    if max_size is None:
        max_size = Config.RESIZE_MAX_EDGE

    h_img, w_img = image.shape[:2]
    if max(h_img, w_img) > max_size:
        scale = max_size / max(h_img, w_img)
        new_w = int(w_img * scale)
        new_h = int(h_img * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
    return tensor, image

def compute_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, Config.HIST_BINS, Config.HIST_RANGES)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist
