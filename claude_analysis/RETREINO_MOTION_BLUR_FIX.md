# 🔧 Retreino V4 — Fix do bug de amplificação no Motion Blur

## 👨‍💻 Contexto

Retreino experimental (`new_model.py`), motivado pela hipótese: o modelo consegue
diferenciar corretamente fotos de **luz pontual dura contra escuridão vasta**
("Distanciamento/Low-key" — silhueta, retrato em contraluz, ambiente escuro com
fonte de luz concentrada)?

Antes de validar essa hipótese, foi feita uma inspeção do dataset
(`dataset/approveds/`, `dataset/failures/`) olhando a distribuição de
`exposure_ratio` (fração de pixels nos extremos do histograma — quase-preto +
quase-branco), já que esse é o indicador mais direto do padrão em questão.

---

## 🔴 Bug encontrado: `apply_motion_blur` amplificava brilho em vez de borrar

**Arquivo:** `dataset/dataset_generator.py`

```python
# ANTES (bug):
def apply_motion_blur(img):
    size = random.randint(8,20)
    kernel = np.zeros((size, size))
    if random.choice([True, False]):
        kernel[int((size-1)/2) :] = np.ones(size)   # ❌ preenche ~metade das LINHAS
    else:
        kernel[:, int((size-1)/2) ] = np.ones(size)  # ✅ preenche 1 coluna (correto)
    kernel /= size
    return cv2.filter2D(img, -1, kernel)
```

No ramo `if`, `kernel[mid:]` faz slice de **linha `mid` até o fim** (todas as
colunas), preenchendo ~metade da matriz com 1s — não uma única linha, como o
ramo `else` faz corretamente para coluna. Dividir por `size` (em vez de pela
contagem real de 1s) deixa a soma do kernel em torno de `size/2` (≈4 a 10) em
vez de 1. Resultado: o filtro não borra, ele **amplifica o brilho** até saturar
em branco (255), independente do conteúdo da foto original.

**Evidência:** `dataset/originals/IMG_1166.jpg` é uma foto de praia normal, dia
claro, sem nada estourado (mean=142). Depois de `apply_motion_blur`, virava uma
imagem quase inteiramente branca. Confirmado visualmente e numericamente:

```
trial 0: mean=305.7 (bug) → mean=142.0 (fix), sem pixels clipados em branco
```

**Fix aplicado:**
```python
kernel[int((size-1)/2), :] = np.ones(size)   # 1 linha, não um bloco de linhas
```

### Impacto no dataset

Antes do fix, **43 imagens** em `dataset/failures/` tinham `exposure_ratio >
0.5` — das quais **25 eram `bad_motion_blur`**, isto é, ruído do bug (fotos de
praia/dia viradas em branco), não corrupção de exposição de verdade. Depois do
fix, regenerando o dataset (`dataset/dataset_generator.py`), caíram para **15**,
dominadas por `bad_exposure` (corrupção de gamma, que é intencional) — a
composição do dataset agora reflete o que o pipeline pretende simular.

| | antes do fix | depois do fix |
|---|---|---|
| `failures/` com `exposure_ratio > 0.5` | 43 | 15 |
| — dos quais `bad_motion_blur` | 25 (ruído do bug) | 1 (motion blur na própria foto low-key) |
| `failures/` mean `exposure_ratio` | 0.288 | 0.176 |

---

## 🟡 Limitação que continua valendo: dataset não valida a hipótese do low-key

Mesmo com o bug corrigido, o dataset **ainda não tem material real para ensinar
ou testar** o padrão "luz dura contra escuridão":

- `dataset/approveds/`: **1 única foto** real nesse perfil de exposição
  (`good_IMG_4146.jpg`, `exposure_ratio=0.61` — luzes penduradas à noite,
  fundo escuro). Um ponto de dado não ensina nada a uma random forest.
- `dataset/failures/`: nenhuma das imagens com `exposure_ratio` alto é uma
  captura genuína de "low-key mal executado" — são corrupções sintéticas
  genéricas (gamma) aplicadas a fotos sem relação com o padrão (jardim ao
  entardecer, etc).

No relatório de métricas abaixo, `good_IMG_4146.jpg` — a única foto aprovada
desse perfil — **continua sendo reprovada pelo modelo novo** (score 0.191).
Isso não é uma regressão introduzida pelo fix; é a confirmação de que o dataset
não tem sinal suficiente para esse padrão específico. Pra testar a hipótese de
verdade, é preciso curar fotos reais (aprovadas e reprovadas) nesse estilo —
o gerador sintético genérico não cobre esse caso.

---

## 📊 Métricas — `new_technical_model.xml` vs `technical_model.xml`

### Comparação "justa" (MAE, holdout 20% real, split idêntico para os dois)

Gerada automaticamente pelo próprio `new_model.py` durante o treino:

```
MAE (modelo novo, fórmula de ratio nova): 0.2722
MAE (modelo antigo, fórmula de ratio antiga): 0.2912
✅ Modelo novo performou MELHOR — candidato à promoção
```

### Classificação no dataset completo (245 imagens: 49 approveds + 196 failures)

Gerado por `metrics/evaluate_metrics.py` (thresholds: aprovado ≥0.65,
reprovado <0.35, revisão entre os dois).

**⚠️ Ressalva:** essa comparação não é apples-to-apples — `technical_model.xml`
foi treinado numa versão anterior do dataset (antes do fix do bug e da
regeneração atual), então parte da diferença abaixo reflete desatualização do
baseline, não só o fix em si. O número mais confiável pra decisão de promoção
é o MAE holdout acima, que usa o mesmo split para os dois.

| Métrica | `technical_model.xml` (antigo) | `new_technical_model.xml` (novo) |
|---|---|---|
| Acurácia | 56.7% (139/245) | **91.8%** (225/245) |
| Precision (Aprovado) | 0.282 | **0.796** |
| Recall (Aprovado) | 0.755 | 0.796 |
| Precision (Reprovado) | 0.895 | **0.949** |
| Recall (Reprovado) | 0.520 | **0.949** |
| Falsos Positivos (aprovou foto ruim) | 94 | **10** |
| Falsos Negativos (reprovou foto boa) | 12 | 10 |
| Casos "Revisão Humana" | 131 (53%!) | 11 (4.5%) |

O modelo antigo jogava mais da metade do dataset pra "revisão humana" — a
zona cinzenta estava absorvendo o erro em vez de classificar. O modelo novo
reduz drasticamente isso.

Relatórios completos salvos em:
- `metrics/reports/new_technical_model_<data>.txt`
- `metrics/reports/old_technical_model_<data>.txt`

---

## 📁 Arquivos alterados nesse ciclo

- `dataset/dataset_generator.py` — fix do bug de `apply_motion_blur`.
- `new_model.py` — `PATH_APPROVEDS` corrigido para `dataset/approveds/` (antes
  apontava pra `dataset/originals/` direto, sem passar pelo split
  approved/failed); removido import morto de `evaluate` (sombreado pela
  definição local, causava import circular).
- `metrics/evaluate_metrics.py` — estava desatualizado (apontava pro
  `technical_model_v2.xml`, que não existe, e usava um vetor de features
  diferente do que o modelo atual foi treinado). Corrigido para reusar
  `extract_raw_metrics`/`assemble_feature_vector` de `new_model.py` — garante
  que o vetor de features usado na avaliação é sempre idêntico ao de
  treinamento, e evita duplicação de lógica que pode dessincronizar de novo.
- `dataset/approveds/`, `dataset/failures/` — regenerados (gitignored,
  reprodutíveis via `uv run python3 dataset/dataset_generator.py`).

## ⏭️ Status

`new_technical_model.xml` **não foi promovido** (não renomeado para
`technical_model.xml`) nem commitado — aguardando decisão. Baseado no MAE
holdout, é candidato à promoção, mas antes disso vale decidir se a frente do
padrão low-key será encerrada ou se o dataset será enriquecido com fotos reais
curadas para esse caso específico.
