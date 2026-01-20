import torch
import cv2
import numpy as np
import gc
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
from kornia.feature import LoFTR
from .config import Config
from .utils import preprocess_tensor

class FeatureExtractor:
    def extract_and_match(self, query_img, sku_img):
        raise NotImplementedError

class LoFTRExtractor(FeatureExtractor):
    def __init__(self):
        self.matcher = LoFTR(pretrained="outdoor").eval().to(Config.DEVICE)

    def extract_and_match(self, query_img, sku_img):
        # LoFTR handles resizing internally usually, but best to provide reasonable input
        query_tensor, query_resized = preprocess_tensor(query_img)
        sku_tensor, sku_resized = preprocess_tensor(sku_img)

        with torch.no_grad():
            input_dict = {"image0": query_tensor, "image1": sku_tensor}
            correspondences = self.matcher(input_dict)
        
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        # Cleanup
        del query_tensor, sku_tensor, correspondences
        torch.cuda.empty_cache()
        gc.collect()
        
        if len(mkpts0) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0
            
        # Rescale points
        scale_x_0 = query_img.shape[1] / query_resized.shape[1]
        scale_y_0 = query_img.shape[0] / query_resized.shape[0]
        scale_x_1 = sku_img.shape[1] / sku_resized.shape[1]
        scale_y_1 = sku_img.shape[0] / sku_resized.shape[0]
        
        mkpts0[:, 0] *= scale_x_0
        mkpts0[:, 1] *= scale_y_0
        mkpts1[:, 0] *= scale_x_1
        mkpts1[:, 1] *= scale_y_1
        
        return mkpts0, mkpts1, len(mkpts0)

class LightGlueExtractor(FeatureExtractor):
    def __init__(self, feature_type="superpoint"):
        self.feature_type = feature_type
        if feature_type == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=Config.MAX_KEYPOINTS).eval().to(Config.DEVICE)
        elif feature_type == "disk":
            self.extractor = DISK(max_num_keypoints=Config.MAX_KEYPOINTS).eval().to(Config.DEVICE)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        self.matcher = LightGlue(features=feature_type).eval().to(Config.DEVICE)

    def extract_and_match(self, query_img, sku_img):
        query_tensor, query_resized = preprocess_tensor(query_img)
        sku_tensor, sku_resized = preprocess_tensor(sku_img)

        with torch.no_grad():
            feats0 = self.extractor.extract(query_tensor)
            feats1 = self.extractor.extract(sku_tensor)
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0 = feats0['keypoints'].cpu().numpy()
        kpts1 = feats1['keypoints'].cpu().numpy()
        matches = matches01['matches'].cpu().numpy()
        
        valid_indices = np.where(matches >= 0)[0]
        
        del feats0, feats1, matches01, query_tensor, sku_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        if len(valid_indices) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0

        scale_x_0 = query_img.shape[1] / query_resized.shape[1]
        scale_y_0 = query_img.shape[0] / query_resized.shape[0]
        scale_x_1 = sku_img.shape[1] / sku_resized.shape[1]
        scale_y_1 = sku_img.shape[0] / sku_resized.shape[0]
        
        mkpts0 = kpts0[valid_indices].copy()
        mkpts1 = kpts1[matches[valid_indices].astype(int)].copy()
        
        mkpts0[:, 0] *= scale_x_0
        mkpts0[:, 1] *= scale_y_0
        mkpts1[:, 0] *= scale_x_1
        mkpts1[:, 1] *= scale_y_1
        
        return mkpts0, mkpts1, len(mkpts0)

class SIFTExtractor(FeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def extract_and_match(self, query_img, sku_img):
        gray0 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(sku_img, cv2.COLOR_BGR2GRAY)
        
        kp0, des0 = self.sift.detectAndCompute(gray0, None)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        
        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0
        
        matches = self.matcher.knnMatch(des0, des1, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0

        mkpts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches])
        mkpts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches])
        
        return mkpts0, mkpts1, len(good_matches)

class ORBExtractor(FeatureExtractor):
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_and_match(self, query_img, sku_img):
        gray0 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(sku_img, cv2.COLOR_BGR2GRAY)
        
        kp0, des0 = self.orb.detectAndCompute(gray0, None)
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        
        if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0
        
        matches = self.matcher.knnMatch(des0, des1, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), 0

        mkpts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches])
        mkpts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches])
        
        return mkpts0, mkpts1, len(good_matches)
