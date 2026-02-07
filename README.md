# Modelo de Avaliação de Qualidade de Imagens
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-green)](https://github.com/astral-sh/uv)

Este projeto implementa um modelo de aprendizado de máquina para avaliação automática da qualidade de imagens. O modelo usa o regressor Random Trees do OpenCV para pontuar imagens em uma escala de 0 (qualidade ruim) a 1 (qualidade excelente) com base em várias características visuais.

## Características

O modelo extrai e analisa as seguintes características das imagens:
- **Nitidez**: Mede a clareza da imagem usando variância laplaciana
- **Densidade de Bordas**: Calcula a porcentagem de pixels de borda usando detecção de bordas Canny
- **Saturação**: Analisa a vivacidade das cores no espaço HSV
- **Contraste**: Mede o desvio padrão dos valores em escala de cinza
- **Exposição**: Avalia sub/super exposição usando análise de histograma
- **Magnitude do Gradiente**: Computa a força das bordas usando operadores Sobel
- **Entropia**: Mede o conteúdo de informação da imagem
- **Faixa Dinâmica**: Analisa o intervalo entre o 5º e 95º percentil das intensidades dos pixels
- **Pontuação de Razão**: Uma métrica composta combinando nitidez e densidade de bordas

## Instalação

### Pré-requisitos
- Python 3.12 ou superior
- Gerenciador de pacotes uv (recomendado) ou pip

### Instalar Dependências
```bash
# Usando uv (recomendado)
uv sync

# Ou usando pip
pip install -r requirements.txt
```

## Uso

### Treinando o Modelo

Para treinar o modelo no seu conjunto de dados:

1. Coloque imagens aprovadas em `dataset/approveds/`
2. Coloque imagens rejeitadas em `dataset/failures/`
3. Execute o script de treinamento:

```bash
uv run python model.py
```

Isso irá gerar `technical_model.xml` contendo o modelo Random Trees treinado.

### Avaliando Imagens

Use o script de avaliação de métricas para testar o modelo:

```bash
uv run python metrics/evaluate_metrics.py
```

### Analisando Características

Para analisar características de imagens individuais:

```bash
uv run python claude_analysis/analyze_features.py
```

## Estrutura do Projeto

```
├── model.py                 # Treinamento principal do modelo e extração de características
├── main.py                  # Ponto de entrada (placeholder)
├── pyproject.toml          # Configuração do projeto e dependências
├── technical_model.xml     # Saída do modelo treinado
├── dataset/
│   ├── approveds/          # Imagens de treinamento de alta qualidade
│   ├── failures/           # Imagens de treinamento de baixa qualidade
│   └── images/             # Imagens de teste
├── metrics/
│   └── evaluate_metrics.py # Avaliação e teste do modelo
├── tests/                  # Testes unitários
└── claude_analysis/        # Documentação de análise e melhorias
    ├── ANALISE_CORRETA.md
    ├── RESUMO_EXECUTIVO.md
    └── ...
```

## Arquitetura do Modelo

O modelo usa o regressor Random Trees (Random Forest) do OpenCV com a seguinte configuração:
- Profundidade Máxima: 28
- Contagem Mínima de Amostras: 3
- Precisão de Regressão: 0.00001
- Iterações Máximas: 300

O treinamento combina imagens reais das pastas do conjunto de dados com dados sintéticos gerados usando intervalos de qualidade predefinidos para cada categoria de característica.

## Melhorias Recentes

- Corrigido viés contra fotos de estúdio com baixa faixa dinâmica
- Adicionado suporte para fotografia minimalista (céus limpos, composições simples)
- Otimizados hiperparâmetros para melhor precisão
- Aprimorada geração de dados sintéticos com categorias ponderadas

## Uso do Claude para Refinamento do Regressor

Este projeto utilizou o Claude Sonnet 4.5 para análise detalhada e refinamento do modelo de regressão. O processo de refinamento incluiu:

### Análise de Problemas Identificados

O Claude foi usado para identificar e corrigir os seguintes problemas principais:

1. **Viés de Faixa Dinâmica**: Fotos de estúdio com DR baixo (5-35) eram incorretamente penalizadas como imagens ruins, quando na verdade representam fotografias de alta qualidade com fundos limpos.

2. **Confusão entre Minimalismo e Imagens Planas**: O modelo confundia fotografias minimalistas válidas (céus limpos, composições simples) com imagens planas de baixa qualidade.

3. **Threshold de Desfoque Permissivo**: Imagens com nitidez muito baixa (< 5) passavam com scores altos, permitindo aprovação de fotos severamente borradas.

4. **Priorização Insuficiente da Exposição**: Fotos com exposição inadequada eram aprovadas mesmo com alta nitidez.

### Correções Implementadas com Claude

**Mudanças nos Intervalos de Características:**
- Adicionado "blur_severe" (0.5-5) para nitidez - sempre reprovado
- Detalhamento da exposição em: good/acceptable/bad_moderate/bad_severe
- Marcação de "studio" (5-35) como excelente para faixa dinâmica

**Expansão dos Dados Sintéticos:**
- Categoria de fotos de estúdio (peso 4x = 520 exemplos)
- Minimalismo extremo (peso 2x = 260 exemplos)
- Exposição tóxica (peso 3x = 390 exemplos)

**Otimização de Hiperparâmetros:**
- Profundidade Máxima: 20 → 28 (+40%)
- Contagem Mínima de Amostras: 5 → 3 (-40%)
- Iterações: 200 → 300 (+50%)

### Resultados Esperados

| Categoria | Antes | Depois |
|-----------|-------|--------|
| Fotos Boas Aprovadas | 10% | 70-80% |
| Fotos Ruins Reprovadas | 60% | 90% |

### Documentação das Análises

Todas as análises e correções realizadas pelo Claude estão documentadas na pasta `claude_analysis/`, incluindo:
- `ANALISE_CORRETA.md`: Diagnóstico técnico detalhado
- `RESUMO_EXECUTIVO.md`: Resumo das melhorias implementadas
- `index.md`: Índice completo das modificações
- Relatórios específicos sobre exposição, minimalismo e correções de blur

## Dependências

- numpy >= 2.4.1
- opencv-python >= 4.11.0.86
- opencv-stubs >= 0.1.1
- requests >= 2.32.5
- scikit-learn >= 1.8.0

## Testes

Execute a suíte de testes:

```bash
uv run python -m pytest tests/
```

## Licença

Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

Copyright (c) 2025 Leo

## Contribuição

1. Faça um fork do repositório
2. Crie uma branch de funcionalidade
3. Faça suas alterações
4. Adicione testes se aplicável
5. Envie um pull request

## Suporte

Para problemas ou dúvidas, abra uma issue no repositório GitHub.

*Feito com ❤️ para processamento de imagens eficiente.*
