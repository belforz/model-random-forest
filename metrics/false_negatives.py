import cv2
import numpy as np
import os

# --- CONFIG ---
PATH_APROVADAS = "dataset/approveds/"   
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

    # Features
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
    
    ratio = np.tanh(sharpness / (edge_density * 50.0 + 1.0))
        
    return [sharpness, edge_density, entropy, mean_magnitude, exposure_ratio, ratio]

def find_the_culprits():
    print(f"--- 🕵️ Find the culprits (Errors) ---")
    
   
    try:
        model = cv2.ml.RTrees_load(MODEL_PATH)
    except:
        print(f"{MODEL_PATH}")
        return

    if not model.isTrained():
        print("Error: Model is not trained.")
        return

    errors_found = 0
    
    print(f"Analyzing folder: {PATH_APROVADAS}...\n")
    print(f"{'FILE':<30} | {'SCORE':<8} | {'SHARPNESS':<8} | {'EDGES':<8} | {'EXP':<6} | {'DIAGNOSIS'}")
    print("-" * 110)

    for f in os.listdir(PATH_APROVADAS):
        if f.lower().endswith(('jpg', 'png', 'jpeg')):
            path = os.path.join(PATH_APROVADAS, f)
            feats = extract_features(path)
            
            if feats:
                # Make the prediction
                sample = np.array([feats], dtype=np.float32)
                raw_score = model.predict(sample)[1][0][0]
                prediction = 1 if raw_score >= 0.5 else 0
                
               
                if prediction == 0:
                    errors_found += 1
                    
                   
                    diagnosis = "Uncertain"
                    s, e, exp, g, ent, r = feats
                    
                    if s < 60: diagnosis = "Very low sharpness (<60)"
                    elif e < 5.0: diagnosis = "Few edges (<5%)"
                    elif exp > 0.3: diagnosis = "Poor exposure (>30%)"
                    elif ent > 7.5: diagnosis = "High entropy/noise"
                    elif g < 15: diagnosis = "Low gradient (<15)"
                    elif r > 0.9: diagnosis = "High sharpness to edge ratio (>0.9)"
                    else: diagnosis = "Complex combination"

                    print(f"{f:<30} | {raw_score:<8.4f} | {s:<8.1f} | {e:<8.2f}% | {exp:<6.2f} | {diagnosis}")

    print("-" * 110)
    print(f"Total False Negatives found: {errors_found}")
    print("Tip: Open the images and review by yourself")

if __name__ == "__main__":
    find_the_culprits()