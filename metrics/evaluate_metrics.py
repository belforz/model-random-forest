import cv2
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
PATH_APPROVEDS = "dataset/approveds/"
PATH_REJECTED = "dataset/failures/"
MODEL_PATH = "technical_model_v2.xml"

# Thresholds otimizados
THRESHOLD_APPROVED = 0.65  # Aumentado de 0.50
THRESHOLD_REJECTED = 0.35  # Novo threshold explícito


def extract_features(image_path):
    """Extrai as 10 features do modelo v2"""
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

        # [8] Saturation Variance
        saturation_var = s_std[0][0] ** 2

        # [9] Dynamic Range
        cdf = hist.cumsum()
        cdf_normalized = cdf * (1.0 / cdf.max())
        p5_idx = np.searchsorted(cdf_normalized, 0.05)
        p95_idx = np.searchsorted(cdf_normalized, 0.95)
        dynamic_range = float(p95_idx - p5_idx)

        # [10] Texture Score (NOVO - substitui o Ratio tóxico)
        texture_score = (sharpness * edge_density) / (1000.0 + exposure_ratio * 5000)
        texture_score = min(texture_score, 10.0)

        return [
            sharpness,
            edge_density,
            saturation_mean,
            contrast_std,
            exposure_ratio,
            mean_magnitude,
            entropy,
            saturation_var,
            dynamic_range,
            texture_score
        ]
    except Exception as e:
        print(f"❌ Erro ao extrair features de {image_path}: {e}")
        return None


def classify_score(score):
    """Classifica o score com thresholds otimizados"""
    if score >= THRESHOLD_APPROVED:
        return "Aprovado", 1
    elif score < THRESHOLD_REJECTED:
        return "Reprovado", 0
    else:
        return "Revisão Humana", 0.5  # Neutro para métricas


def evaluate():
    print("=" * 60)
    print(f"📊 AVALIANDO MODELO V2: {MODEL_PATH}")
    print("=" * 60)
    print(f"Thresholds:")
    print(f"  • Aprovado:  score >= {THRESHOLD_APPROVED}")
    print(f"  • Reprovado: score <  {THRESHOLD_REJECTED}")
    print(f"  • Revisão:   {THRESHOLD_REJECTED} <= score < {THRESHOLD_APPROVED}")
    print("=" * 60)

    # Carregar modelo
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        print(f"Execute primeiro: uv run python3 model_v2_improved.py")
        return

    model = cv2.ml.RTrees_load(MODEL_PATH)
    if not model.isTrained():
        print("❌ Erro: Modelo não carregou corretamente.")
        return

    y_true = []
    y_pred = []
    results_detailed = []

    # Processar Aprovadas (Esperado: 1)
    print("\n🟢 Testando classe 'Aprovadas'...")
    approved_count = 0
    for f in os.listdir(PATH_APPROVEDS):
        if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
            feats = extract_features(os.path.join(PATH_APPROVEDS, f))
            if feats:
                sample = np.array([feats], dtype=np.float32)
                raw_score = model.predict(sample)[1][0][0]
                status, pred_class = classify_score(raw_score)

                y_true.append(1)
                y_pred.append(pred_class)
                
                results_detailed.append({
                    'file': f,
                    'expected': 'Aprovado',
                    'score': raw_score,
                    'status': status,
                    'correct': status == "Aprovado"
                })
                approved_count += 1

    print(f"   Processadas: {approved_count} imagens")

    # Processar Reprovadas (Esperado: 0)
    print("\n🔴 Testando classe 'Reprovadas'...")
    rejected_count = 0
    for f in os.listdir(PATH_REJECTED):
        if f.lower().endswith(('jpg', 'png', 'jpeg', 'webp')):
            feats = extract_features(os.path.join(PATH_REJECTED, f))
            if feats:
                sample = np.array([feats], dtype=np.float32)
                raw_score = model.predict(sample)[1][0][0]
                status, pred_class = classify_score(raw_score)

                y_true.append(0)
                y_pred.append(pred_class)
                
                results_detailed.append({
                    'file': f,
                    'expected': 'Reprovado',
                    'score': raw_score,
                    'status': status,
                    'correct': status == "Reprovado"
                })
                rejected_count += 1

    print(f"   Processadas: {rejected_count} imagens")

    # Relatório Final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE PERFORMANCE")
    print("=" * 60)

    # Confusion Matrix
    # Converter revisão (0.5) para classe mais próxima para matriz
    y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    total = len(y_true)
    correct = tp + tn
    accuracy = (correct / total) * 100
    
    print(f"\n📈 Métricas Gerais:")
    print(f"   Total testado:  {total}")
    print(f"   Corretas:       {correct} ({accuracy:.1f}%)")
    print(f"   Erros:          {fp + fn} ({100-accuracy:.1f}%)")
    print(f"\n🔢 Matriz de Confusão:")
    print(f"   True Negative (TN):  {tn}")
    print(f"   False Positive (FP): {fp}")
    print(f"   False Negative (FN): {fn}")
    print(f"   True Positive (TP):  {tp}")

    # Métricas detalhadas
    print("\n" + "-" * 60)
    print(classification_report(
        y_true, y_pred_binary,
        target_names=['Reprovado (0)', 'Aprovado (1)'],
        digits=3
    ))

    # Análise de erros
    print("\n" + "=" * 60)
    print("🔍 ANÁLISE DETALHADA DE ERROS")
    print("=" * 60)
    
    errors = [r for r in results_detailed if not r['correct']]
    if errors:
        print(f"\n❌ Total de erros: {len(errors)}\n")
        for err in errors[:10]:  # Mostrar primeiros 10
            print(f"   {err['file']}")
            print(f"      Esperado: {err['expected']}")
            print(f"      Obtido:   {err['status']} (score: {err['score']:.3f})")
            print()
    else:
        print("\n🎉 Nenhum erro! Modelo perfeito!")

    # Análise de Revisão Humana
    review_cases = [r for r in results_detailed if r['status'] == "Revisão Humana"]
    print("\n" + "=" * 60)
    print(f"🟡 CASOS DE REVISÃO HUMANA: {len(review_cases)}")
    print("=" * 60)
    
    if review_cases:
        print(f"\nTotal: {len(review_cases)} imagens na zona cinzenta\n")
        for case in review_cases[:5]:
            print(f"   {case['file']}")
            print(f"      Esperado: {case['expected']}")
            print(f"      Score: {case['score']:.3f}")
            print()

    print("\n" + "=" * 60)
    print("✅ AVALIAÇÃO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
