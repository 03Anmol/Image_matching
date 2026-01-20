import cv2
import numpy as np

def visualize_matches(img0, img1, kpts0, kpts1, title="Matches"):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    h_max = max(h0, h1)
    vis = np.zeros((h_max, w0 + w1, 3), dtype=np.uint8)
    vis[:h0, :w0] = img0
    vis[:h1, w0:w0+w1] = img1
    
    if len(kpts0) == 0:
        text = f"{title}: 0 matches"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis
    
    kpts0 = np.array(kpts0).reshape(-1, 2)
    kpts1 = np.array(kpts1).reshape(-1, 2)
    
    num_to_draw = min(len(kpts0), 100)
    for i in range(num_to_draw):
        pt0 = (int(kpts0[i, 0]), int(kpts0[i, 1]))
        pt1 = (int(kpts1[i, 0] + w0), int(kpts1[i, 1]))
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(vis, pt0, 3, color, -1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.line(vis, pt0, pt1, color, 1)
    
    text = f"{title}: {len(kpts0)} matches"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis

class ROISelector:
    def select_roi(self, img):
        print("\n" + "="*70)
        print("INSTRUCTIONS:")
        print("Select a rectangle region for the floor using the mouse.")
        print("Press SPACE/ENTER to confirm selection.")
        print("Press c to cancel selection (skip image).")
        print("="*70 + "\n")

        # cv2.selectROI returns (x, y, w, h)
        # showCrosshair=True, fromCenter=False
        try:
            rect = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI")
        except Exception:
            cv2.destroyAllWindows()
            return None

        x, y, w, h = rect
        
        # If user cancelled (w=0 or h=0)
        if w == 0 or h == 0:
            return None
            
        # Convert to 4 points: TL, TR, BR, BL
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
        
        return pts
