"""
🔬 ANÁLISE COMPARATIVA: Modelo Original vs V2
═══════════════════════════════════════════════

Analisa features de imagens GOOD que foram reprovadas
para identificar padrões e validar as correções.
"""

import cv2
import numpy as np
import os

def extract_features(image_path):
    """Extrai features para análise"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = gray.size

    try:
        # Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F)
        stddev_lap = cv2.meanStdDev(laplacian_var)
        sharpness = stddev_lap[0].item() ** 2

        # Edge Density
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        lower = max(0, mean_val - std_val)
        upper = min(255, mean_val + std_val)
        edges = cv2.Canny(gray, int(lower), int(upper))
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100.0

        # Saturation
        s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
        saturation_mean = s_mean[0][0]
        saturation_var = s_std[0][0] ** 2

        # Contrast
        c_mean, c_std = cv2.meanStdDev(gray)
        contrast_std = c_std[0][0]

        # Exposure
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        exposure_ratio = (np.sum(hist[:31]) + np.sum(hist[225:])) / total_pixels

        # Gradient
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        mean_magnitude = np.mean(magnitude)

        # Entropy
        hist_norm = hist.ravel() / total_pixels
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))

        # Ratio
        ratio_score = np.tanh(sharpness / (edge_density * 50.0 + 1.0))

        # Dynamic Range
        cdf = hist.cumsum()
        cdf_normalized = cdf * (1.0 / cdf.max())
        p5_idx = np.searchsorted(cdf_normalized, 0.05)
        p95_idx = np.searchsorted(cdf_normalized, 0.95)
        dynamic_range = float(p95_idx - p5_idx)

        return {
            'sharpness': sharpness,
            'edge_density': edge_density,
            'saturation_mean': saturation_mean,
            'contrast': contrast_std,
            'exposure': exposure_ratio,
            'gradient': mean_magnitude,
            'entropy': entropy,
            'ratio': ratio_score,
            'saturation_var': saturation_var,
            'dynamic_range': dynamic_range
        }
    except Exception as e:
        print(f"Erro: {e}")
        return None


def analyze_problem_images():
    """Analisa imagens GOOD que foram reprovadas"""
    
    print("=" * 70)
    print("🔬 ANÁLISE DE FEATURES - FOTOS BOAS REPROVADAS")
    print("=" * 70)
    
    # Lista de imagens GOOD reprovadas do output.json
    problem_cases = [
        ("[EXPECT_GOOD]_sample2.jpg", 0.155),
        ("[EXPECT_GOOD]_sample9.jpg", 0.297),
    ]
    
    # Lista de imagens GOOD aprovadas para comparação
    good_cases = [
        ("[EXPECT_GOOD]_sample3.jpg", 0.655),
    ]
    
    base_path = "/home/leo/ai-pre-process-images/images/blind_test/"
    
    print("\n🔴 IMAGENS BOAS QUE FORAM REPROVADAS:")
    print("-" * 70)
    
    problem_features = []
    for filename, score in problem_cases:
        path = base_path + filename
        if os.path.exists(path):
            feats = extract_features(path)
            if feats:
                problem_features.append(feats)
                print(f"\n📷 {filename} (Score: {score})")
                print(f"   Sharpness:      {feats['sharpness']:.2f}")
                print(f"   Edge Density:   {feats['edge_density']:.2f}%")
                print(f"   Contrast:       {feats['contrast']:.2f}")
                print(f"   Exposure:       {feats['exposure']:.4f}")
                print(f"   Saturation:     {feats['saturation_mean']:.2f}")
                print(f"   Dynamic Range:  {feats['dynamic_range']:.2f} ⚠️")
                print(f"   Ratio:          {feats['ratio']:.4f}")
                print(f"   Entropy:        {feats['entropy']:.2f}")
        else:
            print(f"   ❌ Arquivo não encontrado: {path}")
    
    print("\n\n🟢 IMAGENS BOAS QUE FORAM APROVADAS (COMPARAÇÃO):")
    print("-" * 70)
    
    good_features = []
    for filename, score in good_cases:
        path = base_path + filename
        if os.path.exists(path):
            feats = extract_features(path)
            if feats:
                good_features.append(feats)
                print(f"\n📷 {filename} (Score: {score})")
                print(f"   Sharpness:      {feats['sharpness']:.2f}")
                print(f"   Edge Density:   {feats['edge_density']:.2f}%")
                print(f"   Contrast:       {feats['contrast']:.2f}")
                print(f"   Exposure:       {feats['exposure']:.4f}")
                print(f"   Saturation:     {feats['saturation_mean']:.2f}")
                print(f"   Dynamic Range:  {feats['dynamic_range']:.2f}")
                print(f"   Ratio:          {feats['ratio']:.4f}")
                print(f"   Entropy:        {feats['entropy']:.2f}")
        else:
            print(f"   ❌ Arquivo não encontrado: {path}")
    
    # Análise comparativa
    if problem_features and good_features:
        print("\n\n" + "=" * 70)
        print("📊 ANÁLISE COMPARATIVA - MÉDIAS")
        print("=" * 70)
        
        problem_avg = {
            key: np.mean([f[key] for f in problem_features])
            for key in problem_features[0].keys()
        }
        
        good_avg = {
            key: np.mean([f[key] for f in good_features])
            for key in good_features[0].keys()
        }
        
        print("\n┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
        print("│ Feature             │ Reprovadas   │ Aprovadas    │ Diferença    │")
        print("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
        
        for key in ['sharpness', 'edge_density', 'contrast', 'exposure', 
                    'dynamic_range', 'ratio', 'entropy']:
            diff = problem_avg[key] - good_avg[key]
            diff_pct = (diff / good_avg[key] * 100) if good_avg[key] != 0 else 0
            
            marker = "⚠️" if key == 'dynamic_range' and abs(diff_pct) > 50 else ""
            
            print(f"│ {key:19} │ {problem_avg[key]:11.2f}  │ {good_avg[key]:11.2f}  │ {diff:11.2f}  │ {marker}")
        
        print("└─────────────────────┴──────────────┴──────────────┴──────────────┘")
        
        print("\n🔍 INSIGHTS:")
        
        # Dynamic Range
        dr_diff_pct = (problem_avg['dynamic_range'] - good_avg['dynamic_range']) / good_avg['dynamic_range'] * 100
        if dr_diff_pct < -30:
            print(f"\n⚠️  DYNAMIC RANGE: Reprovadas têm DR {abs(dr_diff_pct):.1f}% MENOR")
            print(f"    → Provável causa: Fotos de estúdio/fundo limpo")
            print(f"    → Correção: Modelo V2 dá peso 4x maior para DR baixo + contraste alto")
        
        # Contrast
        if problem_avg['contrast'] > good_avg['contrast'] * 0.7:
            print(f"\n✅ CONTRASTE: Reprovadas têm contraste BOM ({problem_avg['contrast']:.1f})")
            print(f"    → Não é blur, é característica de estúdio")
        
        # Exposure
        if problem_avg['exposure'] < 0.4:
            print(f"\n✅ EXPOSIÇÃO: Reprovadas têm exposição BOA ({problem_avg['exposure']:.3f})")
            print(f"    → Não é problema de iluminação")
    
    print("\n\n" + "=" * 70)
    print("🎯 CONCLUSÃO")
    print("=" * 70)
    print("""
O modelo original penaliza FORTEMENTE Dynamic Range baixo, mas:

1. DR baixo (15-35) + Contraste alto (> 40) = FOTO DE ESTÚDIO EXCELENTE ✅
2. DR baixo + Contraste baixo (< 30) = Flat/Screenshot RUIM ❌

Modelo V2 aprende essa diferença com dados sintéticos balanceados.
""")


if __name__ == "__main__":
    analyze_problem_images()
