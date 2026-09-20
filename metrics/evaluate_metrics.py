import sys
import os
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_model import extract_raw_metrics, assemble_feature_vector

# --- CONFIG ---
PATH_APPROVEDS = "dataset/approveds/"
PATH_REJECTED = "dataset/failures/"
MODEL_PATH = "new_technical_model.xml"
RATIO_FORMULA = "new"  # "new" p/ new_technical_model.xml, "old" p/ technical_model.xml

# Thresholds otimizados
THRESHOLD_APPROVED = 0.65  # Aumentado de 0.50
THRESHOLD_REJECTED = 0.35  # Novo threshold explícito


def extract_features(image_path, ratio_formula=RATIO_FORMULA):
    """Extrai as 10 features usando o mesmo pipeline de new_model.py,
    garantindo que o vetor bate exatamente com o que o modelo foi treinado."""
    raw = extract_raw_metrics(image_path)
    if raw is None:
        return None
    return assemble_feature_vector(raw, ratio_formula)


def classify_score(score):
    """Classifica o score com thresholds otimizados"""
    if score >= THRESHOLD_APPROVED:
        return "Aprovado", 1
    elif score < THRESHOLD_REJECTED:
        return "Reprovado", 0
    else:
        return "Revisão Humana", 0.5  # Neutro para métricas


def evaluate(model_path=MODEL_PATH, ratio_formula=RATIO_FORMULA):
    print("=" * 60)
    print(f"📊 AVALIANDO MODELO: {model_path} (ratio formula: {ratio_formula})")
    print("=" * 60)
    print(f"Thresholds:")
    print(f"  • Aprovado:  score >= {THRESHOLD_APPROVED}")
    print(f"  • Reprovado: score <  {THRESHOLD_REJECTED}")
    print(f"  • Revisão:   {THRESHOLD_REJECTED} <= score < {THRESHOLD_APPROVED}")
    print("=" * 60)

    # Carregar modelo
    if not os.path.exists(model_path):
        print(f"❌ ERRO: Modelo não encontrado em {model_path}")
        return

    model = cv2.ml.RTrees_load(model_path)
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
            feats = extract_features(os.path.join(PATH_APPROVEDS, f), ratio_formula)
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
            feats = extract_features(os.path.join(PATH_REJECTED, f), ratio_formula)
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
