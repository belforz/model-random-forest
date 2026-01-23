

## ------------ SETTINGS ------------ ##
import random
import numpy as np
import cv2
import os


INPUT_FOLDER = "dataset/originals/"
OUTPUT_GOOD = "dataset/approveds/"
OUTPUT_BAD = "dataset/failures/"

TARGET_SIZE = 640

## ---------------------------------- ##

def ensure_directories():
    for f in [OUTPUT_GOOD, OUTPUT_BAD, INPUT_FOLDER]:
        if not os.path.exists(f):
            os.makedirs(f)
            
def resizeSmart(image, max_dim=TARGET_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_motion_blur(img):
    size = random.randint(8,20)
    kernel = np.zeros((size, size))  
    if random.choice([True, False]):
        kernel[int((size-1)/2) :] = np.ones(size)
    else:
        kernel[:, int((size-1)/2) ] = np.ones(size)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def apply_focus_blur(img):
    k = random.choice([5,7,9,11,13,15])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_exposure_error(img):
    gamma = random.choice([0.3,0.4,2.5,3.5])
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_noise(img):
    row, col, ch = img.shape
    mean = 0
    sigma = random.randint(10,30)
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def generate():
    ensure_directories()
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not files:
        print("No images found in the input folder.")
        return
    print(f"Found {len(files)} images. Processing...")
    for fname in files:
        path = os.path.join(INPUT_FOLDER, fname)
        original = cv2.imread(path)
        if original is None: continue
        # save good ones redimensioned
        good_img = resizeSmart(original)
        cv2.imwrite(os.path.join(OUTPUT_GOOD, f"good_{fname}"), good_img)
        # create bad versions
        bad_blur = apply_motion_blur(original)
        cv2.imwrite(os.path.join(OUTPUT_BAD, f"bad_motion_blur_{fname}"), resizeSmart(bad_blur))
        
        bad_motion = apply_focus_blur(good_img)
        cv2.imwrite(os.path.join(OUTPUT_BAD, f"bad_focus_blur_{fname}"), bad_motion)
        
        bad_expo = apply_exposure_error(good_img)
        cv2.imwrite(os.path.join(OUTPUT_BAD, f"bad_exposure_{fname}"), bad_expo)
        
        bad_noise = apply_noise(good_img)
        cv2.imwrite(os.path.join(OUTPUT_BAD, f"bad_noise_{fname}"), bad_noise)
    print(f"✅ Concluded!")
    print(f"Originals processed in: {OUTPUT_GOOD}")
    print(f"Synthetics generated in:    {OUTPUT_BAD}")
    print(f"All resized to max {TARGET_SIZE}px.")
    
if __name__ == "__main__":
    generate()