import cv2
import numpy as np
import os
import random


PATH_APPROVEDS = "dataset/approveds/"
PATH_FAILURES = "dataset/failures/"
OLD_MODEL_PATH = "technical_model.xml"
NEW_MODEL_PATH = "new_technical_model.xml"

REAL_PHOTO_WEIGHT = 10
TEST_SPLIT = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

#ratio

def compute_ratio(sharpness,edge_density,formula="new"):
        if formula == "new":
            return float(np.tanh(np.log1p(sharpness) / (np.log1p(edge_density) * 2.5 + 1.8)))
        else:
            return float(np.tanh(sharpness / (edge_density * 50.0 + 1.0)))
    
#metrics

def extract_raw_metrics(img_path):
    img = cv2.imread(img_path)
    if img is None or img.size == 0:
        print(f"Error: Unable to read image at path {img_path}")
        return None
    h , w = img.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    try:
        total_pixels = gray.size
        #Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F)
        stddev_lap = cv2.meanStdDev(laplacian_var)
        sharpness = stddev_lap[0].item() ** 2
        # Edge density
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        lower = max(0, mean_val - std_val)
        upper = min(255, mean_val + std_val)
        edges = cv2.Canny(gray, int(lower), int(upper))
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100.0
        # Saturation Mean
        s_mean, s_std = cv2.meanStdDev(hsv[:,:,1])
        saturation_mean = s_mean[0][0]
        # Contrast
        _, c_std = cv2.meanStdDev(gray)
        contrast_std = c_std[0][0]
        #Exposure ratio
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        exposure_ratio = (np.sum(hist[:31]) + np.sum(hist[225:])) / total_pixels
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        mean_magnitude = np.mean(magnitude)
        # Entropy
        hist_norm = hist.ravel() / total_pixels
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        # Saturation Variance
        saturation_var = s_std[0][0] ** 2
        # Dynamic Range
        cdf = hist.cumsum()
        cdf_normalized = cdf * (1.0 / cdf.max())
        p5_idx = np.searchsorted(cdf_normalized, 0.05)
        p95_idx = np.searchsorted(cdf_normalized, 0.95)
        dynamic_range = float(p95_idx - p5_idx)
        
        return {
            "sharpness": sharpness,
            "edge_density": edge_density,
            "saturation_mean": saturation_mean,
            "contrast_std": contrast_std,
            "exposure_ratio": exposure_ratio,
            "mean_magnitude": mean_magnitude,
            "entropy": entropy,
            "saturation_var": saturation_var,
            "dynamic_range": dynamic_range
        }
    except Exception as e:
        print(f"Error extracting features at path {img_path}: {e}")
        return None

# Assemble the feature vector from the extracted raw features and the computed ratio score
def assemble_feature_vector(raw, ratio_formula="new"):
    ratio_score = compute_ratio(raw["sharpness"], raw["edge_density"], formula=ratio_formula)
    return [
        raw["sharpness"],
        raw["edge_density"],
        raw["saturation_mean"],
        raw["contrast_std"],
        raw["exposure_ratio"],
        raw["mean_magnitude"],
        raw["entropy"],
        ratio_score,
        raw["saturation_var"],
        raw["dynamic_range"],
       
    ]

def load_real_data():
    entries = []
    for path, is_approved in [(PATH_APPROVEDS, True), (PATH_FAILURES, False)]:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
        for f in os.listdir(path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                raw = extract_raw_metrics(os.path.join(path, f))
                if raw is not None:
                    entries.append((raw, is_approved))
                else:
                    print(f"Failed to extract features for image: {os.path.join(path, f)}")
    n_approved = sum(1 for _, ok in entries if ok)
    n_failures = sum(1 for _, ok in entries if not ok)
    print(f"Number of approved images: {n_approved}")
    print(f"Number of failed images: {n_failures}")
    return entries

def label_for(is_approved):
    return random.uniform(0.75,1.0) if is_approved else random.uniform(0.0,0.25)

def calc_ratio(s,e):
    return compute_ratio(s,e, formula="new")

def _generate_synthetic_rule_data(samples_per_category=150):
    data = []
    labels = []
    print("🔥 Generate sintetic data v2 ...")
    
    def val(metric, category):
        min_val, max_val = RANGES[metric][category]
        return random.uniform(min_val, max_val)
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
        labels.append(random.uniform(0.15, 0.25))
    
    for _ in range(samples_per_category * 2):
        s = random.uniform(0.5, 5)
        e = random.uniform(0, 0.5)
        exp = val('exposure', random.choice(['good', 'acceptable']))
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),
            val('contrast', random.choice(['flat', 'normal', 'good'])),
            exp,
            val('gradient', 'low'),
            val('entropy', random.choice(['low', 'med', 'high'])),
            calc_ratio(s, e),
            random.uniform(0, 1500),
            random.uniform(10, 100),
        ]
        data.append(vec)
        labels.append(random.uniform(0.0, 0.20))
    
    for _ in range(samples_per_category // 2):
        s = val('sharpness', 'low')
        e = random.uniform(0, 1.5)
        exp = val('exposure', 'good')
        vec = [
            s, e,
            random.uniform(0, 80),
            random.uniform(5, 25),
            exp,
            random.uniform(0, 15),
            val('entropy', 'low'),
            calc_ratio(s, e),
            random.uniform(0, 50),
            random.uniform(5, 30),
        ]
        data.append(vec)
        labels.append(random.uniform(0.0, 0.15))
    
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
        
    for _ in range(samples_per_category * 2):
        s = val('sharpness', 'high')
        e = val('edges', 'high')
        exp = val('exposure', 'good')
        vec = [
            s, e,
            val('saturation', random.choice(['normal', 'vibrant'])),
            val('contrast', random.choice(['good', 'high'])),
            exp,
            val('gradient', 'high'),
            val('entropy', random.choice(['med', 'high'])),
            calc_ratio(s, e),
            random.uniform(500, 3000),
            val('dynamic_range', 'studio'),
        ]
        data.append(vec)
        labels.append(random.uniform(0.85, 1.0))
 
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
 
    for _ in range(samples_per_category):
        s = val('sharpness', random.choice(['med', 'high']))
        e = val('edges', random.choice(['low', 'med']))
        exp = val('exposure', 'good')
        vec = [
            s, e,
            val('saturation', 'vibrant'),
            val('contrast', random.choice(['normal', 'good'])),
            exp,
            val('gradient', random.choice(['med', 'high'])),
            val('entropy', 'med'),
            calc_ratio(s, e),
            random.uniform(300, 1500),
            val('dynamic_range', random.choice(['studio', 'normal'])),
        ]
        data.append(vec)
        labels.append(random.uniform(0.70, 0.92))
 
    for _ in range(samples_per_category * 2):
        s = random.uniform(10, 100)
        e = random.uniform(0.2, 2.0)
        exp = val('exposure', 'good')
        vec = [
            s, e,
            val('saturation', random.choice(['bw', 'low', 'normal'])),
            val('contrast', random.choice(['normal', 'good'])),
            exp,
            random.uniform(3, 15),
            val('entropy', random.choice(['med', 'high'])),
            calc_ratio(s, e),
            random.uniform(200, 1200),
            val('dynamic_range', random.choice(['normal', 'high'])),
        ]
        data.append(vec)
        labels.append(random.uniform(0.65, 0.85))
 
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
            val('dynamic_range', 'high'),
        ]
        data.append(vec)
        labels.append(random.uniform(0.75, 0.95))
    print(f"Generated {len(data)} samples with {len(labels)} labels.")
    return data, labels

def build_rtrees():
    rf = cv2.ml.RTrees_create()
    rf.setMaxDepth(28)
    rf.setMinSampleCount(3)
    rf.setRegressionAccuracy(0.00001)
    rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 300, 0.0003))
    rf.setActiveVarCount(0)
    return rf

def train():
    print("=" * 60)
    print(("🧠 Training model V2"))
    print("=" * 60)
    real_entries = load_real_data()
    
    #split real_entries out of training for evaluation
    indices = list(range(len(real_entries)))
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - TEST_SPLIT))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
 
    real_train = [real_entries[i] for i in train_idx]
    test_entries = [real_entries[i] for i in test_idx]
    print(f"Split {len(real_entries)} entries into {len(real_train)} training and {len(test_entries)} testing.")
    real_train_data, real_train_labels = [], []
    for raw, is_approved in real_train:
        vec = assemble_feature_vector(raw, "new")
        for _ in range(REAL_PHOTO_WEIGHT):
            real_train_data.append(vec)
            real_train_labels.append(label_for(is_approved))
 
    synth_data, synth_labels = _generate_synthetic_rule_data(samples_per_category=130)
 
    final_data = real_train_data + synth_data
    final_labels = real_train_labels + synth_labels
    print(f"✅ Total of training samples: {len(final_data)} samples "
          f"({len(real_train_data)} real [{len(real_train)} photos × {REAL_PHOTO_WEIGHT}] "
          f"+ {len(synth_data)} synthetic)")
 
    train_matrix = np.array(final_data, dtype=np.float32)
    labels_matrix = np.array(final_labels, dtype=np.float32)
 
    rf = build_rtrees()
    tdata = cv2.ml.TrainData_create(train_matrix, cv2.ml.ROW_SAMPLE, labels_matrix)
    rf.train(tdata)
    rf.save(NEW_MODEL_PATH)
    print(f"🎉 New model saved at: {NEW_MODEL_PATH} (old model was NOT overwritten)")
 
    evaluate(rf, test_entries)
    
    
def evaluate(new_model, test_entries):
    if not test_entries:
        print("⚠️  No test data — cannot compare. Run with more photos.")
        return
 
    ref_labels = np.array([label_for(ok) for _, ok in test_entries], dtype=np.float32)
 
    new_vectors = np.array(
        [assemble_feature_vector(raw, "new") for raw, _ in test_entries], dtype=np.float32
    )
    _, new_preds = new_model.predict(new_vectors)
    new_preds = new_preds.flatten()
    new_mae = float(np.mean(np.abs(new_preds - ref_labels)))
 
    print("\n" + "=" * 60)
    print("📊 EVALUATION — holdout real, each model with the ratio formula it knows")
    print("=" * 60)
    print(f"MAE (new model, new ratio formula):    {new_mae:.4f}")
 
    if os.path.exists(OLD_MODEL_PATH):
        old_model = cv2.ml.RTrees_load(OLD_MODEL_PATH)
        old_vectors = np.array(
            [assemble_feature_vector(raw, "old") for raw, _ in test_entries], dtype=np.float32
        )
        _, old_preds = old_model.predict(old_vectors)
        old_preds = old_preds.flatten()
        old_mae = float(np.mean(np.abs(old_preds - ref_labels)))
 
        print(f"MAE (old model, old ratio formula): {old_mae:.4f}")
        print("-" * 60)
        if new_mae < old_mae:
            print(f"✅ New model performed BETTER (MAE {new_mae:.4f} < {old_mae:.4f}).")
            print(f"   Consider promoting: rename {NEW_MODEL_PATH} → {OLD_MODEL_PATH}.")
        else:
            print(f"🛑 New model performed SAME OR WORSE (MAE {new_mae:.4f} >= {old_mae:.4f}).")
            print(f"   Keep the current version ({OLD_MODEL_PATH}) — do not promote.")
    else:
        print(f"⚠️  Old model not found at {OLD_MODEL_PATH} — no baseline to compare.")
 
    print("\n📋 Test samples with the largest error (new model):")
    errors = np.abs(new_preds - ref_labels)
    worst_idx = np.argsort(errors)[-5:][::-1]
    for i in worst_idx:
        print(f"  expected={ref_labels[i]:.3f}  predicted={new_preds[i]:.3f}  error={errors[i]:.3f}")
 
 
if __name__ == "__main__":
    train()
    


