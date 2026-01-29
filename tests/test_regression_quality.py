import cv2
import numpy as np
import os

MODEL_PATH = "./technical_model.xml"

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    h, w = img.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F)
    stddev = cv2.meanStdDev(laplacian_var)
    sharpness = stddev[0].item() ** 2
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    lower = max(0, mean_val - std_val)
    upper = min(255, mean_val + std_val)
    edges = cv2.Canny(gray, int(lower), int(upper))
    edge_density = (np.count_nonzero(edges) / total_pixels) * 100.0
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    exposure_ratio = (np.sum(hist[:31]) + np.sum(hist[225:])) / total_pixels
    
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mean_magnitude = np.mean(cv2.magnitude(grad_x, grad_y))
    
    hist_norm = hist.ravel() / total_pixels
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    
    ratio = np.tanh(sharpness / (edge_density * 50.0 + 1.0))
    
    return [sharpness, edge_density, entropy, mean_magnitude, exposure_ratio, ratio]

def test_quality_distribution():
    print("🧪 Testing Regression Quality Distribution")
    print("=" * 80)
    
    model = cv2.ml.RTrees_load(MODEL_PATH)
    
    approved_scores = []
    rejected_scores = []
    
    # Test approved images
    approved_path = "dataset/approveds/"
    if os.path.exists(approved_path):
        for f in os.listdir(approved_path):
            if f.lower().endswith(('jpg', 'png', 'jpeg')):
                feats = extract_features(os.path.join(approved_path, f))
                if feats:
                    sample = np.array([feats], dtype=np.float32)
                    score = model.predict(sample)[1][0][0]
                    approved_scores.append(score)
    
    # Test rejected images
    rejected_path = "dataset/failures/"
    if os.path.exists(rejected_path):
        for f in os.listdir(rejected_path):
            if f.lower().endswith(('jpg', 'png', 'jpeg')):
                feats = extract_features(os.path.join(rejected_path, f))
                if feats:
                    sample = np.array([feats], dtype=np.float32)
                    score = model.predict(sample)[1][0][0]
                    rejected_scores.append(score)
    
    # Statistics
    print(f"\n📊 APPROVED IMAGES (n={len(approved_scores)}):")
    print(f"   Mean:   {np.mean(approved_scores):.4f}")
    print(f"   Median: {np.median(approved_scores):.4f}")
    print(f"   Std:    {np.std(approved_scores):.4f}")
    print(f"   Min:    {np.min(approved_scores):.4f}")
    print(f"   Max:    {np.max(approved_scores):.4f}")
    
    # Distribution buckets
    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    hist_approved, _ = np.histogram(approved_scores, bins=bins)
    print(f"\n   Distribution:")
    print(f"   [0.0-0.2):  {hist_approved[0]:3d} images")
    print(f"   [0.2-0.4):  {hist_approved[1]:3d} images")
    print(f"   [0.4-0.5):  {hist_approved[2]:3d} images")
    print(f"   [0.5-0.6):  {hist_approved[3]:3d} images")
    print(f"   [0.6-0.8):  {hist_approved[4]:3d} images")
    print(f"   [0.8-1.0]:  {hist_approved[5]:3d} images")
    
    print(f"\n📊 REJECTED IMAGES (n={len(rejected_scores)}):")
    print(f"   Mean:   {np.mean(rejected_scores):.4f}")
    print(f"   Median: {np.median(rejected_scores):.4f}")
    print(f"   Std:    {np.std(rejected_scores):.4f}")
    print(f"   Min:    {np.min(rejected_scores):.4f}")
    print(f"   Max:    {np.max(rejected_scores):.4f}")
    
    hist_rejected, _ = np.histogram(rejected_scores, bins=bins)
    print(f"\n   Distribution:")
    print(f"   [0.0-0.2):  {hist_rejected[0]:3d} images")
    print(f"   [0.2-0.4):  {hist_rejected[1]:3d} images")
    print(f"   [0.4-0.5):  {hist_rejected[2]:3d} images")
    print(f"   [0.5-0.6):  {hist_rejected[3]:3d} images")
    print(f"   [0.6-0.8):  {hist_rejected[4]:3d} images")
    print(f"   [0.8-1.0]:  {hist_rejected[5]:3d} images")
    
    print("\n" + "=" * 80)
    print("✅ QUALITY METRICS:")
    
    # Separation
    separation = np.mean(approved_scores) - np.mean(rejected_scores)
    print(f"   Separation (mean diff):     {separation:.4f}")
    
    # Overlap
    overlap = len([s for s in approved_scores if s < 0.5]) + len([s for s in rejected_scores if s >= 0.5])
    print(f"   Overlap (misclassified):    {overlap}")
    
    # Variance check
    total_variance = np.var(approved_scores + rejected_scores)
    print(f"   Total variance:             {total_variance:.4f}")
    
    # Regression quality
    if np.std(approved_scores) > 0.05 and np.std(rejected_scores) > 0.05:
        print(f"\n✅ Model shows good regression behavior (varied scores)")
    else:
        print(f"\n⚠️  Model shows limited regression behavior (scores too clustered)")
    
    if separation > 0.6:
        print(f"✅ Excellent class separation ({separation:.2f})")
    elif separation > 0.4:
        print(f"✅ Good class separation ({separation:.2f})")
    else:
        print(f"⚠️  Weak class separation ({separation:.2f})")

if __name__ == "__main__":
    test_quality_distribution()
