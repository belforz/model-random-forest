import os
import requests
import cv2
import numpy as np
import shutil


OUTPUT_DIR = "images/blind_test_batch/"
TOTAL_SAMPLES = 10  

def download_random_image(index):
    """Baixa uma imagem aleatória de alta qualidade (Picsum)"""
    url = "https://picsum.photos/640/480" 
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            path = os.path.join(OUTPUT_DIR, f"temp_{index}.jpg")
            with open(path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            return path
    except Exception as e:
        print(f"❌ Erro ao baixar imagem: {e}")
    return None

def create_variants(img_path, index):
    img = cv2.imread(img_path)
    if img is None: return

   
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"[EXPECT_GOOD]_sample{index}.jpg"), img)

   
    blur = cv2.GaussianBlur(img, (25, 25), 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"[EXPECT_BAD_BLUR]_sample{index}.jpg"), blur)

   
    overexposed = cv2.convertScaleAbs(img, alpha=2.5, beta=50)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"[EXPECT_BAD_EXPOSURE]_sample{index}.jpg"), overexposed)

    
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    vintage = cv2.add(img, noise)
    
    vintage = cv2.GaussianBlur(vintage, (3, 3), 0) 
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"[EXPECT_VINTAGE]_sample{index}.jpg"), vintage)

    
    os.remove(img_path)

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"🌍 Baixando {TOTAL_SAMPLES} imagens inéditas da internet...")
    
    for i in range(TOTAL_SAMPLES):
        print(f"   ⬇️ Baixando amostra {i+1}/{TOTAL_SAMPLES}...")
        path = download_random_image(i)
        if path:
            create_variants(path, i)

    print("\n✅ LOTE DE TESTE CEGO GERADO!")
    print(f"📁 Pasta: {os.path.abspath(OUTPUT_DIR)}")
    print("👉 Agora rode seu C++ apontando para esta pasta e verifique se os prefixos batem com o resultado.")

if __name__ == "__main__":
    main()