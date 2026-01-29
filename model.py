import cv2
import numpy as np
import os
import random

PATH_APPROVEDS = "dataset/approveds/"
PATH_FAILURES = "dataset/failures/"
MODEL_OUTPUT = "technical_model.xml"

RANGES = {
    'sharpness': {'low': (0, 300), 'med': (301, 1500) , 'high': (1501, 15000)},
    'edges':     {'low': (0, 4.9), 'med': (5.0, 15.0) , 'high': (15.1, 50.0)},
    'entropy':   {'low': (0, 4.5), 'med': (4.6, 7.0) , 'high': (7.1, 10.0)},
    'gradient':  {'low': (0, 25), 'med': (26, 60) , 'high': (61, 200)},
    'exposure':  {'ok': (0.0, 0.40), 'bad': (0.42, 1.0)}
}

def extract_features_from_image(img):
    """Extract features from a given image."""
    img = cv2.imread(img)
    if img is None or img.size == 0:
        print(f"Error reading image: {img}")
        return None
    
    h, w = img.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return None
    
    total_pixels = gray.size
    
    try:
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
        
        # Ratio Tanh
        
        ratio_score = np.tanh(sharpness / (edge_density * 50.0 + 1.0))
        
        
        return [sharpness, edge_density, entropy, mean_magnitude, exposure_ratio, ratio_score]
    except Exception as e:
        print(f"Error processing image {img}: {e}")
        return None

def _generate_synthetic_rule_data(samples_per_rule=(None, 80)):
    """Generate synthetic data based on predefined rules."""
    data = []
    labels = []
    print(f"Generating {samples_per_rule} synthetic data...")
    
    def val(metric, value):
        min_val, max_val = RANGES[metric][value]
        return random.uniform(min_val, max_val)
    
    def calc_ratio(s, e):
        return np.tanh(s / (e * 50.0 + 1.0))
    
   
    # 1. Toxic Brightness
    for _ in range(samples_per_rule[1]):
        s, e = val('sharpness', 'high'), val('edges', 'med')
        vec = [s, e, val('exposure', 'bad'), val('gradient', 'high'), random.uniform(4,9), calc_ratio(s,e)]
        data.append(vec); labels.append(0.0)

    # 2. Blur
    for _ in range(samples_per_rule[1]):
        s, e = val('sharpness', 'low'), val('edges', 'low')
        vec = [s, e, val('exposure', 'ok'), val('gradient', 'low'), val('entropy', 'high'), calc_ratio(s,e)]
        data.append(vec); labels.append(0.0)

    # 3. Fake Landscape 
    for _ in range(samples_per_rule[1]):
        s, e = val('sharpness', 'low'), val('edges', 'low')
        vec = [s, e, val('exposure', 'ok'), val('gradient', 'low'), val('entropy', 'med'), calc_ratio(s,e)]
        data.append(vec); labels.append(0.0)

    
    # 4. True Landscape
    for _ in range(samples_per_rule[1]):
        s, e = val('sharpness', 'med'), val('edges', 'low')
        vec = [s, e, val('exposure', 'ok'), val('gradient', 'low'), val('entropy', 'low'), calc_ratio(s,e)]
        data.append(vec); labels.append(1.0)

    # 5. Generic Good
    for _ in range(samples_per_rule[1] * 2):
        s, e = val('sharpness', 'high'), val('edges', 'med')
        vec = [s, e, val('exposure', 'ok'), val('gradient', 'high'), val('entropy', 'med'), calc_ratio(s,e)]
        data.append(vec); labels.append(1.0)
    
    return data, labels

def train():
    print("----- Initializing hibrid model training( Imagess + Logical Rules) ----")
    final_data = []
    final_labels = []
    print("📂 Processing images and generating synthetic data...")
    count_imgs = 0
    real_data_multiplier = 15
    
    if os.path.exists(PATH_APPROVEDS):
        for f in os.listdir(PATH_APPROVEDS):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                features = extract_features_from_image(os.path.join(PATH_APPROVEDS, f))
                if features and all(isinstance(feat, (int, float, np.number)) for feat in features):
                    for _ in range(real_data_multiplier):
                        final_data.append(features)
                        final_labels.append(1)  # APPROVED
                        count_imgs += 1
                else:
                    print(f"Skipped approved image {f}: features={features}")
    
    if os.path.exists(PATH_FAILURES):
        for f in os.listdir(PATH_FAILURES):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                features = extract_features_from_image(os.path.join(PATH_FAILURES, f))
                if features and all(isinstance(feat, (int, float, np.number)) for feat in features):
                    for _ in range(real_data_multiplier):   
                        final_data.append(features)
                        final_labels.append(0)  # REPROVED
                        count_imgs += 1
                else:
                    print(f"Skipped failure image {f}: features={features}")
    print(f"✅ Processed {count_imgs} images from dataset folders.")
    synth_data, synth_labels = _generate_synthetic_rule_data(samples_per_rule=(None, 80))
    final_data.extend(synth_data)
    final_labels.extend(synth_labels)
    print(f"✅ Generated {len(synth_data)} synthetic data based on rules. ")
    
    train_matrix = np.array(final_data, dtype=np.float32)
    labels_matrix = np.array(final_labels, dtype=np.int32)
    
    print("🤖 Dataset is mounted: ")
    print(f" - Real Imasges: {count_imgs}")
    print(f" - Synthetic Data: {len(synth_data)}")
    print(f" - Total Samples: {len(final_data)}")
    print("⚙️ Training the model...")
    
    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(12)
    rf.setMinSampleCount(5)
    rf.setRegressionAccuracy(0.0001)
    rf.setMaxCategories(2)
    
    tdata = cv2.ml.TrainData_create(train_matrix, cv2.ml.ROW_SAMPLE, labels_matrix)
    rf.train(tdata)
    rf.save(MODEL_OUTPUT)
    
    print(f"✅ Model V3 trained and saved to {MODEL_OUTPUT}")
    print(f"📊 Total samples used: {len(final_data)}")
    print(f"📊 Features per sample: {len(final_data[0]) if final_data else 0}")
    print("🎉 Training completed successfully!")
    
if __name__ == "__main__":
    train()
    
    
    
    