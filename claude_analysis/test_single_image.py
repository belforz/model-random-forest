"""
🧪 TESTE DE IMAGEM ESPECÍFICA
═══════════════════════════════════════════════════════════════════

Testa uma única imagem e mostra todas as features + predição.
Útil para debug de casos específicos.
"""

import cv2
import numpy as np
import sys
import os

MODEL_PATH = "technical_model.xml"

def extract_features(image_path):
    """Extrai as 10 features"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Erro ao carregar: {image_path}")
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
        c_mean, c_std = cv2.meanStdDev(gray)
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

        return {
            'vector': [
                sharpness, edge_density, saturation_mean, contrast_std,
                exposure_ratio, mean_magnitude, entropy, ratio_score,
                saturation_var, dynamic_range
            ],
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
        print(f"❌ Erro ao extrair features: {e}")
        return None


def classify_score(score):
    """Classifica o score"""
    if score >= 0.65:
        return "✅ APROVADO", "🟢"
    elif score < 0.35:
        return "❌ REPROVADO", "🔴"
    else:
        return "⚠️ REVISÃO HUMANA", "🟡"


def analyze_profile(feats):
    """Analisa o perfil da imagem"""
    profiles = []
    
    # Minimalismo Extremo
    if (feats['sharpness'] < 100 and feats['edge_density'] < 2.0 and
        feats['contrast'] > 35 and feats['entropy'] > 6.5):
        profiles.append("🎨 MINIMALISMO EXTREMO")
    
    # Foto de Estúdio
    if (feats['sharpness'] > 1500 and feats['contrast'] > 50 and
        feats['dynamic_range'] < 40):
        profiles.append("📸 FOTO DE ESTÚDIO")
    
    # Blur
    if feats['sharpness'] < 500 and feats['edge_density'] < 3 and feats['contrast'] < 35:
        profiles.append("🌫️ BLUR/FLAT")
    
    # Exposição Ruim
    if feats['exposure'] > 0.5:
        profiles.append("💡 EXPOSIÇÃO RUIM")
    
    # Alta Complexidade
    if feats['entropy'] > 7.5 and feats['edge_density'] > 15:
        profiles.append("🏙️ ALTA COMPLEXIDADE")
    
    return profiles if profiles else ["📷 FOTO NORMAL"]


def test_image(image_path):
    """Testa uma imagem"""
    print("=" * 70)
    print("🧪 TESTE DE IMAGEM ESPECÍFICA")
    print("=" * 70)
    print(f"\n📷 Imagem: {os.path.basename(image_path)}")
    print(f"   Path: {image_path}\n")
    
    # Verificar se modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo não encontrado: {MODEL_PATH}")
        print(f"   Execute: uv run python3 model_v2_fixed.py")
        return
    
    # Verificar se imagem existe
    if not os.path.exists(image_path):
        print(f"❌ Imagem não encontrada: {image_path}")
        return
    
    # Extrair features
    print("🔍 Extraindo features...")
    feats = extract_features(image_path)
    
    if not feats:
        print("❌ Falha ao extrair features.")
        return
    
    # Mostrar features
    print("\n" + "─" * 70)
    print("📊 FEATURES EXTRAÍDAS")
    print("─" * 70)
    
    print(f"\n{'Feature':<20} {'Valor':>12}  {'Categoria'}")
    print("─" * 70)
    
    # Sharpness
    s_cat = "🔴 Muito Baixa" if feats['sharpness'] < 500 else "🟡 Média" if feats['sharpness'] < 2000 else "🟢 Alta"
    print(f"{'Sharpness':<20} {feats['sharpness']:>12.2f}  {s_cat}")
    
    # Edge Density
    e_cat = "🔴 Muito Baixa" if feats['edge_density'] < 3 else "🟡 Média" if feats['edge_density'] < 12 else "🟢 Alta"
    print(f"{'Edge Density':<20} {feats['edge_density']:>11.2f}%  {e_cat}")
    
    # Contrast
    c_cat = "🔴 Baixo" if feats['contrast'] < 30 else "🟡 Normal" if feats['contrast'] < 66 else "🟢 Alto"
    print(f"{'Contrast':<20} {feats['contrast']:>12.2f}  {c_cat}")
    
    # Exposure
    exp_cat = "🟢 Boa" if feats['exposure'] < 0.35 else "🟡 Aceitável" if feats['exposure'] < 0.50 else "🔴 Ruim"
    print(f"{'Exposure Ratio':<20} {feats['exposure']:>12.4f}  {exp_cat}")
    
    # Entropy
    ent_cat = "🔴 Baixa" if feats['entropy'] < 4.5 else "🟡 Média" if feats['entropy'] < 7 else "🟢 Alta"
    print(f"{'Entropy':<20} {feats['entropy']:>12.2f}  {ent_cat}")
    
    # Dynamic Range
    dr_cat = "🟢 Estúdio" if feats['dynamic_range'] < 35 else "🟡 Normal" if feats['dynamic_range'] < 100 else "⚠️ Alto"
    print(f"{'Dynamic Range':<20} {feats['dynamic_range']:>12.2f}  {dr_cat}")
    
    print(f"{'Gradient':<20} {feats['gradient']:>12.2f}")
    print(f"{'Ratio Score':<20} {feats['ratio']:>12.4f}")
    print(f"{'Saturation Mean':<20} {feats['saturation_mean']:>12.2f}")
    print(f"{'Saturation Var':<20} {feats['saturation_var']:>12.2f}")
    
    # Perfil
    print("\n" + "─" * 70)
    print("🏷️ PERFIL DA IMAGEM")
    print("─" * 70)
    profiles = analyze_profile(feats)
    for profile in profiles:
        print(f"   {profile}")
    
    # Carregar modelo e prever
    print("\n" + "─" * 70)
    print("🤖 PREDIÇÃO DO MODELO")
    print("─" * 70)
    
    model = cv2.ml.RTrees_load(MODEL_PATH)
    if not model.isTrained():
        print("❌ Erro ao carregar modelo.")
        return
    
    sample = np.array([feats['vector']], dtype=np.float32)
    raw_score = model.predict(sample)[1][0][0]
    
    status, emoji = classify_score(raw_score)
    
    print(f"\n{emoji} Score: {raw_score:.4f}")
    print(f"   Status: {status}")
    
    # Comparação
    print("\n" + "─" * 70)
    print("📈 INTERPRETAÇÃO")
    print("─" * 70)
    
    if raw_score >= 0.65:
        print("✅ Imagem aprovada automaticamente")
    elif raw_score < 0.35:
        print("❌ Imagem reprovada automaticamente")
    else:
        print("⚠️ Requer revisão humana (zona cinzenta)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: uv run python3 test_single_image.py <caminho_da_imagem>")
        print("\nExemplo:")
        print("  uv run python3 test_single_image.py /home/leo/ai-pre-process-images/images/blind_test/[EXPECT_GOOD]_sample2.jpg")
        sys.exit(1)
    
    test_image(sys.argv[1])
