# Future Improvements 🚀

This document outlines how to take the Floor Matching System from a prototype to a production-grade, scalable, and highly accurate solution.

## 1. Improving Accuracy 🎯

### A. Automatic Segmentation (No More Manual Box!)
Instead of asking the user to draw a box, use AI to find the floor automatically.
- **Tools**: `Segment Anything Model (SAM)` or `Mask2Former`.
- **Benefit**: Fully automated experience; removes user error in selection.

### B. Geometric Verification
Currently, we count matches. We should verify if those matches geometrically make sense (i.e., do they lie on the same plane?).
- **Technique**: Use **RANSAC** (Random Sample Consensus) to find a valid Homography matrix. If the matches don't fit the matrix, discard them.
- **Benefit**: Filters out "lucky" bad matches, significantly reducing false positives.

### C. Advanced Deep Learning Models
- **LoFTR (Detector-Free)**: Works better on smooth floors with low texture where point-based methods (SuperPoint) fail.
- **Metric Learning (Siamese Networks)**: Train a custom ResNet/ViT model specifically on flooring datasets to learn embeddings where similar floors are close in vector space.

---

## 2. Improving Scalability 📈

### A. Pre-computation (The "Offline" Phase)
Currently, we process every SKU every time. This is slow if you have 10,000 products.
- **Solution**: Run feature extraction on all 20,000 SKUs **once** and save them.
- **Format**: Save feature vectors to disk (using `.pt` or `.npy` files).
- **Benefit**: "Zero" latency for SKU processing during a query.

### B. Vector Databases
For massive scale (millions of products), linear scanning is too slow.
- **Solution**: Use a Vector DB like **Milvus**, **FAISS**, or **ChromaDB**.
- **Workflow**: Convert the floor image to a single 512-dim vector -> Query the DB for the nearest neighbors.
- **Benefit**: Search millions of products in milliseconds.

---

## 3. Improving Robustness 🛡️

### A. Lighting Invariance
Floors look different at night vs. day.
- **Technique**: Apply **Histogram Equalization** (CLAHE) or White Balance correction before processing.
- **Benefit**: Matches the *material*, not the *shadows*.

### B. Multi-Scale Matching
Some product images are zoomed out; others are close-ups.
- **Technique**: Create an "image pyramid" (resize the query to 0.5x, 1x, 2x) and match against all scales.
- **Benefit**: Finds matches regardless of how far the camera was from the floor.

### C. Outlier Rejection
Sometimes the "best" match is still a bad match (e.g., if the product isn't in the catalog).
- **Logic**: Set a strict threshold. If the best score is `< 0.3`, return "No Match Found" instead of a wrong guess.
