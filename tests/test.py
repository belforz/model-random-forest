import cv2
import numpy as np
import os
import random


MODEL_PATH = "./technical_model.xml"

# Função auxiliar para calcular o ratio exatamente como no treino
def calc_ratio(s, e):
    return np.tanh(s / (e * 50.0 + 1.0))

def generate_ambiguous_samples(samples_per_type=20):
    """Gera vetores de features projetados para cair na zona de incerteza do modelo."""
    data = []
    labels = [] # Apenas para log visual

    print(f"🧪 Gerando {samples_per_type*3} amostras sintéticas de 'Zona Cinza'...")

    # TIPO 1: A Fronteira da Exposição (0.50 a 0.55)
    # O modelo não foi treinado com nada aqui. É terra de ninguém.
    for _ in range(samples_per_type):
        s = random.uniform(2000, 5000) # Nitidez boa
        e = random.uniform(15, 25)     # Borda normal
        # O PULO DO GATO: Exposição no limbo entre as regras
        exp = random.uniform(0.5001, 0.5499) 
        grad = random.uniform(60, 120)
        ent = random.uniform(6, 7)
        vec = [s, e, exp, grad, ent, calc_ratio(s,e)]
        data.append(vec)
        labels.append("Borderline Exposure (0.50-0.55)")

    # TIPO 2: Conflito de Titãs (Nitidez Alta vs Entropia Alta)
    # Nitidez diz "Aprova", Entropia diz "Reprova (Ruído)"
    for _ in range(samples_per_type):
        s = random.uniform(10000, 15000) # Nitidez EXTREMA (Bom)
        e = random.uniform(40, 50)       # Borda alta
        exp = random.uniform(0.2, 0.4)
        grad = random.uniform(150, 200)
        ent = random.uniform(9.0, 10.0)  # Entropia EXTREMA (Ruim/Ruído)
        vec = [s, e, exp, grad, ent, calc_ratio(s,e)]
        data.append(vec)
        labels.append("Conflict: High Sharp vs High Noise")

    # TIPO 3: O Falso Desfoque (Nitidez Baixa vs Entropia Limpa)
    # Nitidez diz "Reprova (Blur)", Entropia diz "Aprova (Clean)"
    for _ in range(samples_per_type):
        s = random.uniform(50, 200)    # Nitidez MUITO BAIXA (Ruim)
        e = random.uniform(1, 5)       # Borda baixa
        exp = random.uniform(0.2, 0.4)
        grad = random.uniform(10, 30)
        ent = random.uniform(2.0, 3.5) # Entropia MUITO BAIXA (Bom/Limpo)
        vec = [s, e, exp, grad, ent, calc_ratio(s,e)]
        data.append(vec)
        labels.append("Conflict: Low Sharp vs Clean Entropy")

    return np.array(data, dtype=np.float32), labels

def test_regression_capability():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erro: Modelo {MODEL_PATH} não encontrado. Treine primeiro.")
        return

    # 1. Carrega o modelo existente (Regressor V15)
    rf = cv2.ml.RTrees_load(MODEL_PATH)
    print(f"✅ Modelo Regressor carregado: {MODEL_PATH}")

    # 2. Gera os dados difíceis
    X_ambiguous, type_labels = generate_ambiguous_samples(samples_per_type=15) # 45 amostras total

    # 3. Roda a predição
    print("\n⚡ Rodando inferência nas amostras da 'Zona do Crepúsculo'...")
    # O predict retorna a média das árvores (o score float)
    _, preds = rf.predict(X_ambiguous)
    scores = preds.flatten()

    # 4. Mostra os resultados
    print("\n========================================")
    print("📊 RESULTADOS DO TESTE DE REGRESSÃO")
    print("Esperamos ver scores quebrados (ex: 0.45, 0.62, 0.31)")
    print("========================================")
    
    binary_count = 0
    gray_zone_count = 0

    for i, score in enumerate(scores):
        label = type_labels[i]
        
        # Formatação visual para destacar a zona cinza
        prefix = "⚪" # Binário
        if 0.01 < score < 0.99:
            prefix = "🟣 ZONA CINZA"
            gray_zone_count += 1
        else:
            binary_count += 1
            
        print(f"{prefix} | Score: {score:.6f} | Tipo: {label}")

    print("\n========================================")
    print("📈 ESTATÍSTICAS FINAIS")
    print(f"Total de Amostras Testadas: {len(scores)}")
    print(f"Scores Binários (0.0 ou 1.0): {binary_count}")
    print(f"Scores na Zona Cinza (Quebrados): {gray_zone_count}")
    
    if gray_zone_count > 0:
        print("\n✅ SUCESSO: O modelo PROVOU ser um Regressor capaz de incerteza.")
        print(f"Score Mínimo: {np.min(scores):.6f}")
        print(f"Score Máximo: {np.max(scores):.6f}")
        print(f"Desvio Padrão dos Scores: {np.std(scores):.6f} (Maior que 0 indica variação)")
    else:
        print("\n❌ FALHA: O modelo ainda está se comportando como classificador binário.")

if __name__ == "__main__":
    test_regression_capability()