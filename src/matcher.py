import cv2
import numpy as np
from .utils import compute_color_histogram
from .config import Config

class Matcher:
    def __init__(self, extractors):
        self.extractors = extractors

    def compute_color_candidates(self, query_img, sku_paths, top_k=5):
        hist_query = compute_color_histogram(query_img)
        
        candidates = []
        for sku_path in sku_paths:
            sku_img = cv2.imread(sku_path)
            if sku_img is None:
                continue
            
            hist_sku = compute_color_histogram(sku_img)
            score = cv2.compareHist(hist_query, hist_sku, cv2.HISTCMP_CORREL)
            
            if score >= 0.5:
                candidates.append((sku_path, score, sku_img))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
        
    def match_features_for_candidate(self, query_img, sku_img, color_score):
        feature_results = []
        
        h_sku, w_sku = sku_img.shape[:2]
        query_resized = cv2.resize(query_img, (w_sku, h_sku))
        
        best_algo_score = 0
        best_algo_data = None
        
        for name, extractor in self.extractors.items():
            try:
                mkpts0, mkpts1, num_matches = extractor.extract_and_match(query_resized, sku_img)
                
                feature_score = min(num_matches / 100.0, 1.0)
                combined_score = 0.25 * color_score + 0.75 * feature_score
                
                result_entry = {
                    'algorithm': name,
                    'matches': num_matches,
                    'mkpts0': mkpts0,
                    'mkpts1': mkpts1,
                    'combined_score': combined_score
                }
                feature_results.append(result_entry)
                
            except Exception as e:
                print(f"  {name}: Error - {str(e)}")
                
        return feature_results
