import os
import cv2
import glob
import time
import gc
import torch
from src.config import Config
from src.utils import load_image, extract_region_ultra_quality
from src.features import LightGlueExtractor, SIFTExtractor, ORBExtractor, LoFTRExtractor
from src.matcher import Matcher
from src.visualization import ROISelector, visualize_matches

def main():
    Config.ensure_dirs()
    print("Initializing Floor Matching System...")
    
    print("Loading models...")
    extractors = {
        "LightGlue-SuperPoint": LightGlueExtractor("superpoint"),
        "LightGlue-DISK": LightGlueExtractor("disk"),
        "LoFTR": LoFTRExtractor(),
        "SIFT": SIFTExtractor(),
        "ORB": ORBExtractor()
    }
    matcher = Matcher(extractors)
    
    query_paths = sorted(glob.glob(os.path.join(Config.QUERY_DIR, "*")))
    sku_paths = sorted(glob.glob(os.path.join(Config.SKU_DIR, "*")))
    
    if not query_paths:
        print(f"No query images found in {Config.QUERY_DIR}")
        return
    if not sku_paths:
        print(f"No SKU images found in {Config.SKU_DIR}")
        return

    print(f"Found {len(query_paths)} queries and {len(sku_paths)} SKUs.")
    
    roi_selector = ROISelector()

    for q_idx, q_path in enumerate(query_paths):
        print(f"\nProcessing Query {q_idx+1}/{len(query_paths)}: {os.path.basename(q_path)}")
        query_img = load_image(q_path)
        if query_img is None:
            print("Failed to load image.")
            continue
            
        pts = roi_selector.select_roi(query_img)
        if pts is None:
            print("Skipping...")
            continue
            
        print("\nExtracting region with Ultra Quality (2.5x upscale + LANCZOS4)...")
        warped = extract_region_ultra_quality(query_img, pts)
        print(f"Extracted region size: {warped.shape[1]}x{warped.shape[0]}")
        
        cv2.imshow("Ultra Quality Extraction", warped)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        print("\nStep 1: Color-based filtering...")
        start_time = time.time()
        candidates = matcher.compute_color_candidates(warped, sku_paths)
        print(f"Found {len(candidates)} color matches")
        
        print("\nStep 2: Feature matching...")
        
        best_overall_match = None
        best_overall_score = 0
        best_algorithm = None
        best_combined_score = 0
        all_results = []
        best_algo_data = {}
        
        for sku_path, color_score, sku_img in candidates[:5]:
            print(f"\nProcessing: {os.path.basename(sku_path)} (color: {color_score:.3f})")
            
            sku_results = matcher.match_features_for_candidate(warped, sku_img, color_score)
            
            for res in sku_results:
                algo_name = res['algorithm']
                num_matches = res['matches']
                combined_score = res['combined_score']
                
                print(f"  {algo_name}: {num_matches} matches")
                
                if combined_score > best_combined_score and num_matches >= 10:
                    best_overall_score = num_matches
                    best_overall_match = sku_path
                    best_algorithm = algo_name
                    best_combined_score = combined_score
                    
                    best_algo_data = {
                        'mkpts0': res['mkpts0'].copy(),
                        'mkpts1': res['mkpts1'].copy(),
                        'sku_img': sku_img.copy(),
                        'query_img_resized': cv2.resize(warped, (sku_img.shape[1], sku_img.shape[0]))
                    }
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            all_results.append({
                'sku_path': sku_path,
                'color_score': color_score,
                'algorithms': sku_results
            })

        print(f"\nMatching completed in {time.time() - start_time:.2f}s")

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        for result in all_results:
            print(f"\n{os.path.basename(result['sku_path'])} (Color: {result['color_score']:.3f})")
            for algo in result['algorithms']:
                print(f"  {algo['algorithm']:<25} {algo['matches']:>4} matches  (combined: {algo['combined_score']:.3f})")
        
        if best_overall_match:
            print("\n" + "="*80)
            print("BEST MATCH")
            print("="*80)
            print(f"File: {os.path.basename(best_overall_match)}")
            print(f"Algorithm: {best_algorithm}")
            print(f"Feature matches: {best_overall_score}")
            print(f"Combined score: {best_combined_score:.3f}")
            print("="*80)
            
            vis = visualize_matches(
                best_algo_data['query_img_resized'], 
                best_algo_data['sku_img'], 
                best_algo_data['mkpts0'], 
                best_algo_data['mkpts1'],
                f"Best - {best_algorithm}"
            )
            
            cv2.imshow("Query (Extracted Region)", warped)
            cv2.imshow("Best Match (SKU)", best_algo_data['sku_img'])
            cv2.imshow("Feature Matches", vis)
            cv2.waitKey(0)
        else:
            print("\nNo suitable match found")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
