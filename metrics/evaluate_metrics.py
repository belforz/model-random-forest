import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
PATH_APPROVEDS = "dataset/approveds/"
PATH_REJECTED = "dataset/failures/"
MODEL_PATH = "technical_model.xml"

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

     # Sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F)
    stddev = cv2.meanStdDev(laplacian_var)
    sharpness = stddev[0].item() ** 2
        
        # Edges
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    lower_threshold = max(0, mean_val - std_val)
    upper_threshold = min(255, mean_val + std_val)
    edges = cv2.Canny(gray, int(lower_threshold), int(upper_threshold))
    edge_density = (np.count_nonzero(edges) / total_pixels) * 100.0
        
        # Exposure
    hist = cv2.calcHist([gray], [0] , None, [256], [0, 256])
    exposure_ratio = (np.sum(hist[:31]) + np.sum(hist[225:])) / total_pixels
        
        # Gradient
    grad_x  = cv2.Sobel(gray, cv2.CV_32F, 1, 0 , ksize=3)
    grad_y  = cv2.Sobel(gray, cv2.CV_32F, 0, 1 , ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    mean_magnitude = np.mean(magnitude)
        
        # Entropy
    hist_norm = hist.ravel() / total_pixels
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
        
        # Ratio
    ratio = sharpness / (edge_density + 1e-5)
        
    return [sharpness, edge_density, entropy, mean_magnitude, exposure_ratio, ratio]

def evaluate():
    print(f"--- 📊 Evaluating Model: {MODEL_PATH} ---")
    
    # Load the model
    model = cv2.ml.RTrees_load(MODEL_PATH)
    if not model.isTrained():
        print("Error: Model did not load correctly.")
        return

    y_true = []
    y_pred = []

    # Process Approved (Expected: 1)
    print("Testing class 'Approved'...")
    for f in os.listdir(PATH_APPROVEDS):
        if f.lower().endswith(('jpg', 'png', 'jpeg')):
            feats = extract_features(os.path.join(PATH_APPROVEDS, f))
            if feats:
                # OpenCV prediction requires samples as float32 numpy arrays    
                sample = np.array([feats], dtype=np.float32)
                prediction = model.predict(sample)[0] 
                
                y_true.append(1)
                y_pred.append(int(prediction))

    # Process Rejected (Expected: 0)
    print("Testing class 'Rejected'...")
    for f in os.listdir(PATH_REJECTED):
        if f.lower().endswith(('jpg', 'png', 'jpeg')):
            feats = extract_features(os.path.join(PATH_REJECTED, f))
            if feats:
                sample = np.array([feats], dtype=np.float32)
                prediction = model.predict(sample)[0]
                
                y_true.append(0)
                y_pred.append(int(prediction))

    # Final Report
    print("\n" + "="*40)
    print("TECHNICAL PERFORMANCE REPORT")
    print("="*40)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Total Tested: {len(y_true)}")
    print(f"Correct: {tp + tn} | Errors: {fp + fn}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    print("-" * 40)
    
    # Detailed Metrics
    print(classification_report(y_true, y_pred, target_names=['Rejected (0)', 'Approved (1)']))
if __name__ == "__main__":
    evaluate()