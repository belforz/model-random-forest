# 🚨 CORREÇÃO CRÍTICA: Threshold Absoluto de Sharpness

## ❌ Problema Descoberto

Fotos com BLUR SEVERO passando com scores altos:

```
[EXPECT_BAD_BLUR]_sample0: 0.60 (ERA 0.16) ❌
[EXPECT_BAD_BLUR]_sample1: 0.53 (ERA 0.16) ❌
[EXPECT_BAD_BLUR]_sample2: 0.61 (ERA 0.16) ❌
[EXPECT_BAD_BLUR]_sample7: 0.64 (ERA 0.18) ❌
```

## 🔍 Root Cause

Features das fotos ruins:
```
Sharpness:      1-3    ← EXTREMAMENTE BAIXA (< 5)
EdgeDensity:    0%     ← LITERALMENTE ZERO
Contrast:       43-73  ← OK/ALTO
Entropy:        7.2-7.7 ← OK/ALTO
```

**Modelo interpretava:** "Contrast OK + Entropy OK = Minimalismo" ✅  
**Realidade:** "Sharpness < 5 = BLUR SEVERO" ❌

## ⚖️ Diferenciação Correta

### ✅ Minimalismo Válido (torre contra céu)
```
Sharpness:      10-100  ← Baixa mas EXISTE algo nítido
EdgeDensity:    0.2-2%  ← Poucas mas EXISTEM
Contrast:       > 35
Entropy:        > 6.5
Exposure:       < 0.35
```
**Exemplo:** Torre em foco, céu desfocado (intencional)

### ❌ Blur Severo (tudo desfocado)
```
Sharpness:      < 5     ← QUASE ZERO (tudo blur)
EdgeDensity:    < 0.5%  ← ZERO/QUASE ZERO
Contrast:       Qualquer
Entropy:        Qualquer
```
**Exemplo:** Foto toda tremida/desfocada

## 🔧 Correções Implementadas

### 1. Threshold Absoluto
```python
# RANGES atualizados:
"sharpness": {
    "blur_severe": (0.5, 5),    # ⚠️ SEMPRE REPROVA
    "low": (5, 500),            # Baixa (pode ser minimalismo)
    "med": (501, 2000),
    "high": (2001, 15000)
}
```

### 2. Minimalismo: Sharpness > 10
```python
# ANTES (ERRADO):
s = random.uniform(5, 100)  # Incluía blur severo

# DEPOIS (CORRETO):
s = random.uniform(10, 100)  # Exclui blur severo
```

### 3. Blur Severo: Peso 2x
```python
# Categoria com PRIORIDADE MÁXIMA
for _ in range(samples_per_category * 2):
    s = random.uniform(0.5, 5)       # Sharpness < 5
    e = random.uniform(0, 0.5)       # Edges ~ 0
    
    # Pode ter contrast/entropy OK (não salva!)
    vec = [..., val('contrast', 'good'), ..., val('entropy', 'high')]
    
    labels.append(random.uniform(0.0, 0.20))  # SEMPRE < 0.20
```

## 📊 Resultados Esperados

### EXPECT_BAD_BLUR (sharpness 1-3):
- **Antes da correção:** 0.50-0.64 (passando) ❌
- **Depois da correção:** 0.05-0.20 (reprovado) ✅

### Minimalismo (sharpness 10-100):
- **Mantém:** 0.65-0.85 (aprovado) ✅

### Fotos de Estúdio (sharpness > 1500):
- **Mantém:** 0.85-1.0 (aprovado) ✅

## 🎯 Regra Final

```
SE sharpness < 5:
    ENTÃO score < 0.20  # SEMPRE REPROVA
SENÃO SE sharpness < 100 E edges < 2%:
    SE contrast > 35 E entropy > 6.5:
        ENTÃO score = 0.65-0.85  # Minimalismo
    SENÃO:
        ENTÃO score < 0.20  # Blur/Flat
SENÃO:
    [outras regras...]
```

## ✅ Teste de Validação

```bash
# Retreinar
uv run python3 model_v2_fixed.py

# Verificar EXPECT_BAD_BLUR
# Esperado: TODOS com score < 0.30
```

---

**Status:** Correção crítica aplicada. Modelo agora diferencia blur severo (sharpness < 5) de minimalismo (sharpness 10-100).
