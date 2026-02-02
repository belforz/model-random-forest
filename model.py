import cv2
import numpy as np
import os
import random

PATH_APPROVEDS = "dataset/approveds/"
PATH_FAILURES = "dataset/failures/"
MODEL_OUTPUT = "technical_model.xml"


RANGES = {
    "sharpness": {
        "blur_severe": (0.5, 5),      
        "low": (5, 500),               
        "med": (501, 2000),            
        "high": (2001, 15000)         
    },
    "edges": {"low": (0, 3.0), "med": (3.1, 12.0), "high": (12.1, 50.0)},
    "entropy": {"low": (0, 4.5), "med": (4.6, 7.0), "high": (7.1, 10.0)},
    "gradient": {"low": (0, 30), "med": (31, 80), "high": (81, 200)},
    "exposure": {
        "good": (0.0, 0.35),           
        "acceptable": (0.36, 0.50),    
        "bad_moderate": (0.51, 0.65),  
        "bad_severe": (0.66, 1.0)      
    },
    "saturation": {"bw": (0, 15), "low": (16, 50), "normal": (51, 120), "vibrant": (121, 255)},
    "contrast": {
        "flat": (0, 30),
        "normal": (31, 65),
        "good": (66, 100),
        "high": (101, 150),
    },
    "dynamic_range": {
        "studio": (5, 35),      
        "normal": (36, 100),    
        "high": (101, 255)      
    }
}


def extract_features_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None or img.size == 0:
        print(f"Error reading image: {img_path}")
        return None

    h, w = img.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    try:
        total_pixels = gray.size

        # [1] Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F)
        stddev_lap = cv2.meanStdDev(laplacian_var)
        sharpness = stddev_lap[0].item() ** 2

        # [2] Edge Density
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        lower = max(0, mean_val - std_val)
        upper = min(255, mean_val + std_val)
        edges = cv2.Canny(gray, int(lower), int(upper))
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100.0

        # [3] Saturation Mean
        s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
        saturation_mean = s_mean[0][0]

        # [4] Contrast
        _, c_std = cv2.meanStdDev(gray)
        contrast_std = c_std[0][0]

        # [5] Exposure Ratio
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        exposure_ratio = (np.sum(hist[:31]) + np.sum(hist[225:])) / total_pixels

        # [6] Gradient Magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        mean_magnitude = np.mean(magnitude)

        # [7] Entropy
        hist_norm = hist.ravel() / total_pixels
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))

        # [8] Ratio Score
        ratio_score = np.tanh(sharpness / (edge_density * 50.0 + 1.0))

        # [9] Saturation Variance
        saturation_var = s_std[0][0] ** 2

        # [10] Dynamic Range
        cdf = hist.cumsum()
        cdf_normalized = cdf * (1.0 / cdf.max())
        p5_idx = np.searchsorted(cdf_normalized, 0.05)
        p95_idx = np.searchsorted(cdf_normalized, 0.95)
        dynamic_range = float(p95_idx - p5_idx)

        return [
            sharpness,          # 1
            edge_density,       # 2
            saturation_mean,    # 3
            contrast_std,       # 4
            exposure_ratio,     # 5
            mean_magnitude,     # 6
            entropy,            # 7
            ratio_score,        # 8 
            saturation_var,     # 9
            dynamic_range,      # 10
        ]

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def _generate_synthetic_rule_data(samples_per_category=150):
    """Improved synthetic data - FOCUS: correct DR bias"""
    data = []
    labels = []
    
    print(f"🔥 Generating optimized synthetic data...")
    
    def val(metric, category):
        min_val, max_val = RANGES[metric][category]
        return random.uniform(min_val, max_val)
    
    def calc_ratio(s, e):
        return np.tanh(s / (e * 50.0 + 1.0))

    # ═══════════════════════════════════════════════════════════════
    # 🔴 FAILED (Score: 0.0 - 0.25)
    # ═══════════════════════════════════════════════════════════════
    
    # 1A. EXTREMELY BAD EXPOSURE (exposure > 0.65)
    # Even with high sharpness/edges, ALWAYS fails
    for _ in range(samples_per_category * 3):  # TRIPLE WEIGHT
        s = random.uniform(100, 12000)  # Can have VERY high sharpness
        e = random.uniform(0, 30)  # Can have VERY high edges
        exp = random.uniform(0.65, 1.0)  # ⚠️ TOXIC EXPOSURE (> 0.65)
        
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),
            val('contrast', random.choice(['flat', 'normal', 'good', 'high'])),  # Any
            exp,  # ← DOMINANT FEATURE
            random.uniform(10, 150),  # Any gradient
            val('entropy', random.choice(['low', 'med', 'high'])),  # Any
            calc_ratio(s, e),
            random.uniform(0, 3000),
            random.uniform(5, 200),  # Any DR
        ]
        data.append(vec)
        labels.append(random.uniform(0.0, 0.15))  # VERY low score
    
    # 1B. MODERATE BAD EXPOSURE (0.50 < exposure < 0.65)
    for _ in range(samples_per_category):
        s = random.uniform(100, 8000)
        e = val('edges', random.choice(['low', 'med', 'high']))
        exp = random.uniform(0.50, 0.65)
        
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),
            val('contrast', random.choice(['flat', 'normal'])),
            exp,
            random.uniform(10, 100),
            val('entropy', random.choice(['low', 'med'])),
            calc_ratio(s, e),
            random.uniform(0, 1000),
            random.uniform(5, 200),
        ]
        data.append(vec)
        labels.append(random.uniform(0.15, 0.25))  # Low score
    
    # 2. SEVERE BLUR (sharpness almost ZERO)
    for _ in range(samples_per_category * 2):  # Double weight - very important
        s = random.uniform(0.5, 5)  # ⚠️ EXTREMELY low sharpness
        e = random.uniform(0, 0.5)  # EdgeDensity ZERO or almost
        exp = val('exposure', random.choice(['good', 'acceptable']))
        
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),
            val('contrast', random.choice(['flat', 'normal', 'good'])),  # Can have contrast
            exp,
            val('gradient', 'low'),
            val('entropy', random.choice(['low', 'med', 'high'])),  # Can have entropy
            calc_ratio(s, e),
            random.uniform(0, 1500),
            random.uniform(10, 100),  # Any DR
        ]
        data.append(vec)
        labels.append(random.uniform(0.0, 0.20))  # ALWAYS failed
    
    # 3. FLAT/SCREENSHOT (different from minimalism!)
    # If low sharpness + low edges + LOW ENTROPY + LOW CONTRAST = BAD
    for _ in range(samples_per_category // 2):
        s = val('sharpness', 'low')
        e = random.uniform(0, 1.5)
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            random.uniform(0, 80),
            random.uniform(5, 25),  # ⚠️ LOW CONTRAST (key difference)
            exp,
            random.uniform(0, 15),
            val('entropy', 'low'),  # ⚠️ LOW ENTROPY (key difference)
            calc_ratio(s, e),
            random.uniform(0, 50),
            random.uniform(5, 30),
        ]
        data.append(vec)
        labels.append(random.uniform(0.0, 0.15))

    # ═══════════════════════════════════════════════════════════════
    # 🟡 GRAY ZONE (Score: 0.30 - 0.50)
    # ═══════════════════════════════════════════════════════════════
    
    # 4. VINTAGE/STYLIZED
    for _ in range(samples_per_category):
        s = val('sharpness', 'med')
        e = val('edges', 'med')
        exp = val('exposure', 'acceptable')
        
        vec = [
            s, e,
            val('saturation', random.choice(['low', 'normal'])),
            val('contrast', 'normal'),
            exp,
            val('gradient', 'med'),
            val('entropy', 'med'),
            calc_ratio(s, e),
            random.uniform(100, 800),
            val('dynamic_range', 'normal'),
        ]
        data.append(vec)
        labels.append(random.uniform(0.35, 0.48))
    
    # 5. SLIGHTLY BLURRY
    for _ in range(samples_per_category // 2):
        s = val('sharpness', 'med')
        e = val('edges', 'low')
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            val('saturation', 'vibrant'),
            val('contrast', 'normal'),
            exp,
            val('gradient', 'med'),
            val('entropy', 'med'),
            calc_ratio(s, e),
            random.uniform(500, 1500),
            val('dynamic_range', 'normal'),
        ]
        data.append(vec)
        labels.append(random.uniform(0.30, 0.45))

    # ═══════════════════════════════════════════════════════════════
    # 🟢 APPROVED (Score: 0.65 - 1.0)
    # ═══════════════════════════════════════════════════════════════
    
    # 6. STUDIO PHOTOS (Low DR = EXCELLENT!)
    # ✅ CRITICAL CORRECTION
    for _ in range(samples_per_category * 4):  # MAX WEIGHT
        s = val('sharpness', 'high')
        e = val('edges', 'high')
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            val('saturation', random.choice(['normal', 'vibrant'])),
            val('contrast', random.choice(['good', 'high'])),  # High contrast
            exp,
            val('gradient', 'high'),
            val('entropy', random.choice(['med', 'high'])),
            calc_ratio(s, e),
            random.uniform(500, 3000),
            val('dynamic_range', 'studio'),  # ✅ LOW DR = GOOD!
        ]
        data.append(vec)
        labels.append(random.uniform(0.85, 1.0))
    
    # 7. NORMAL QUALITY PHOTOS
    for _ in range(samples_per_category * 2):
        s = val('sharpness', 'high')
        e = val('edges', random.choice(['med', 'high']))
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            val('saturation', random.choice(['normal', 'vibrant'])),
            val('contrast', random.choice(['normal', 'good'])),
            exp,
            val('gradient', random.choice(['med', 'high'])),
            val('entropy', random.choice(['med', 'high'])),
            calc_ratio(s, e),
            random.uniform(800, 2500),
            val('dynamic_range', 'normal'),
        ]
        data.append(vec)
        labels.append(random.uniform(0.70, 0.95))
    
    # 8. NORMAL MINIMALISM (low complexity but intentional)
    for _ in range(samples_per_category):
        s = val('sharpness', random.choice(['med', 'high']))
        e = val('edges', random.choice(['low', 'med']))
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            val('saturation', 'vibrant'),  # Color saves
            val('contrast', random.choice(['normal', 'good'])),  # Contrast saves
            exp,
            val('gradient', random.choice(['med', 'high'])),
            val('entropy', 'med'),
            calc_ratio(s, e),
            random.uniform(300, 1500),
            val('dynamic_range', random.choice(['studio', 'normal'])),
        ]
        data.append(vec)
        labels.append(random.uniform(0.70, 0.92))
    
    # 8B. EXTREME MINIMALISM (clear sky, pole, etc)
    # ✅ NEW CATEGORY: Low sharpness but valid photo
    # ⚠️ IMPORTANT: Sharpness must be > 10 (not < 5 which is severe blur)
    for _ in range(samples_per_category * 2):  # Double weight
        s = random.uniform(10, 100)  # Low sharpness but NOT extreme
        e = random.uniform(0.2, 2.0)  # EdgeDensity almost ZERO
        exp = val('exposure', 'good')
        
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),  # Can be light blue sky
            val('contrast', random.choice(['normal', 'good'])),  # Contrast saves
            exp,  # GOOD exposure saves
            random.uniform(3, 15),  # Very low gradient
            val('entropy', random.choice(['med', 'high'])),  # Entropy saves (has information)
            calc_ratio(s, e),  # High ratio (low s/e)
            random.uniform(200, 1200),
            val('dynamic_range', random.choice(['normal', 'high'])),  # Medium/high DR
        ]
        data.append(vec)
        labels.append(random.uniform(0.65, 0.85))  # Good but not perfect
    
    # 9. HIGH COMPLEXITY
    for _ in range(samples_per_category):
        s = val('sharpness', 'high')
        e = val('edges', 'high')
        exp = val('exposure', random.choice(['good', 'acceptable']))
        
        vec = [
            s, e,
            val('saturation', random.choice(['normal', 'vibrant'])),
            val('contrast', random.choice(['normal', 'good', 'high'])),
            exp,
            val('gradient', 'high'),
            val('entropy', 'high'),
            calc_ratio(s, e),
            random.uniform(1000, 3000),
            val('dynamic_range', 'high'),  # High DR OK here
        ]
        data.append(vec)
        labels.append(random.uniform(0.75, 0.95))

    print(f"✅ Generated {len(data)} synthetic examples.")
    return data, labels


def train():
    """Trains the optimized v2 model"""
    print("=" * 60)
    print("🧠 TRAINING MODEL V2 - OPTIMIZED")
    print("=" * 60)
    
    final_data = []
    final_labels = []
    count_imgs = 0

   
    for path, label_range in [
        (PATH_APPROVEDS, (0.75, 1.0)),
        (PATH_FAILURES, (0.0, 0.25)),
    ]:
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    features = extract_features_from_image(os.path.join(path, f))

                    if (
                        features
                        and len(features) == 10
                        and all(isinstance(n, (int, float, np.number)) for n in features)
                    ):
                        
                        for _ in range(10):
                            final_data.append(features)
                            final_labels.append(random.uniform(*label_range))
                        count_imgs += 1
                    else:
                        print(f"⚠️ Ignored {f}: invalid vector.")

    print(f"✅ Loaded {count_imgs} real images.")

    
    synth_data, synth_labels = _generate_synthetic_rule_data(samples_per_category=130)
    final_data.extend(synth_data)
    final_labels.extend(synth_labels)

    print(f"✅ Total of {len(final_data)} samples.")

    
    train_matrix = np.array(final_data, dtype=np.float32)
    labels_matrix = np.array(final_labels, dtype=np.float32)

    print(f"\n⚙️ Configuring Random Trees...")
    
    # Create rf
    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(28)  
    rf.setMinSampleCount(3)  
    rf.setRegressionAccuracy(0.00001)
    rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.0003))
    rf.setActiveVarCount(0)

    # Train
    tdata = cv2.ml.TrainData_create(train_matrix, cv2.ml.ROW_SAMPLE, labels_matrix)
    rf.train(tdata)
    rf.save(MODEL_OUTPUT)

    print(f"\n🎉 Model saved in: {MODEL_OUTPUT}")
    print("=" * 60)
    print("\n📊 Test with: uv run python3 metrics/evaluate_metrics.py\n")


if __name__ == "__main__":
    train()

