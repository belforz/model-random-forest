import cv2
import numpy as np
from typing import Tuple

def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega dados de treinamento simulados para o modelo de qualidade de imagem.
    Features: [Nitidez (Variance), Densidade de Bordas (%), Gradiente Magnitude, Entropia]
    Labels: 1 = Aprovado, 0 = Reprovado
    """
    # Dados de Treino (Matriz Float32)
    # Coluna 0: Nitidez (Variance), Coluna 1: Bordas (Density %), Coluna 2: Gradiente, Coluna 3: Entropia
    train_data = np.array([
        [150.0, 15.0, 50.0, 7.0],  # Ótima -> Aprovado
        [120.0, 13.0, 45.0, 6.5],  # Boa -> Aprovado
        [100.0, 10.0, 40.0, 6.0],  # Boa -> Aprovado
        [80.0, 8.0, 35.0, 5.5],    # Zona Cinza -> Aprovado
        [60.0, 6.0, 30.0, 5.0],    # Zona Cinza -> Reprovado
        [50.0, 4.0, 25.0, 4.5],    # Ruim -> Reprovado
        [20.0, 1.0, 15.0, 3.0],    # Péssima -> Reprovado
        [200.0, 20.0, 60.0, 8.0],  # Excelente -> Aprovado
        [30.0, 2.0, 20.0, 3.5],    # Muito ruim -> Reprovado
        [90.0, 9.0, 38.0, 5.8],    # Aceitável -> Aprovado
    ], dtype=np.float32)

    # Rótulos (Inteiros)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1], dtype=np.int32)

    return train_data, labels

def create_and_train_model(train_data: np.ndarray, labels: np.ndarray) -> cv2.ml.RTrees:
    """
    Cria e treina um modelo Random Forest usando OpenCV.
    """
    # Criar Random Forest
    rf = cv2.ml.RTrees_create()

    # Configurações do modelo
    rf.setMaxDepth(10)
    rf.setMinSampleCount(2)
    rf.setRegressionAccuracy(0)
    rf.setUseSurrogates(False)
    rf.setMaxCategories(2)
    rf.setPriors(np.zeros(0))
    rf.setCalculateVarImportance(True)
    rf.setActiveVarCount(0)

    # Critério de parada
    term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01)
    rf.setTermCriteria(term_crit)

    # Preparar dados para treinamento
    train_data_wrapper = cv2.ml.TrainData_create(train_data, cv2.ml.ROW_SAMPLE, labels)

    # Treinar o modelo
    rf.train(train_data_wrapper)

    return rf

def test_model(rf: cv2.ml.RTrees, test_samples: np.ndarray) -> np.ndarray:
    """
    Testa o modelo com amostras de teste usando 4 features.
    """
    predictions = []
    for sample in test_samples:
        pred = rf.predict(sample.reshape(1, -1))
        predictions.append(int(pred[0]))
    return np.array(predictions)

def main():
    print("Carregando dados de treinamento...")
    train_data, labels = load_training_data()

    print("Treinando modelo...")
    rf = create_and_train_model(train_data, labels)

    # Teste rápido com algumas amostras
    test_samples = np.array([
        [140.0, 14.0, 48.0, 6.8],  # Deve aprovar
        [40.0, 3.0, 22.0, 4.0],    # Deve reprovar
        [70.0, 7.0, 32.0, 5.2],    # Zona cinza
    ], dtype=np.float32)

    print("Testando modelo...")
    predictions = test_model(rf, test_samples)
    for i, pred in enumerate(predictions):
        status = "Aprovado" if pred == 1 else "Reprovado"
        print(f"Amostra {i+1}: {status}")

    # Salvar modelo
    output_file = "technical_model.xml"
    rf.save(output_file)
    print(f"Modelo salvo como '{output_file}'. Pronto para uso no C++!")

if __name__ == "__main__":
    main()