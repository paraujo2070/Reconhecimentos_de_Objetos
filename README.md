# AgroVision: Reconhecimento de Milho e Ervas-Daninhas 🌽🌿

Este projeto implementa um sistema de visão computacional e machine learning para identificação automatizada de plantas de milho e ervas-daninhas em ambiente agrícola. O sistema utiliza técnicas de processamento de imagem clássica (ExG) para segmentação e o algoritmo **Random Forest** para classificação baseada em características geométricas.

## 🚀 Diferenciais do Projeto
- **Sem Deep Learning:** Roda em hardware modesto sem necessidade de GPU.
- **Data Augmentation:** Sistema integrado de rotação para aumentar o dataset.
- **Pipeline:** Estrutura modular separando processamento, extração e treinamento.
- **Relatórios Gerados:** Exportação de dados em CSV para treinamento.

---

## 📂 Estrutura do Projeto

```text
Reconhecimentos_de_Objetos/
├── data/
│   ├── raw/                # Fotos originais separadas por pastas (milho/erva_daninha)
│   ├── processed/          # Imagens após limpeza de solo (Visual)
│   └── output/             # Modelo treinado (.pkl) e relatórios (.csv)
├── src/
│   ├── core/
│   │   ├── processor.py    # Algoritmo ExG para isolar o verde
│   │   └── extractor.py    # Extração de métricas (Circularidade, Solidez, etc)
│   ├── detection/
│   │   └── detector.py     # Segmentação de múltiplas plantas em fotos de campo
│   ├── main.py             # Orquestrador para gerar o dataset de treino
│   └── predict_visual.py   # Script de predição e visualização
├── ml_training/
│   └── train_model.py      # Treinamento da Inteligência Artificial
```

---

## 🛠️ Pré-requisitos

Certifique-se de ter o Python 3.x instalado. As principais bibliotecas utilizadas são:
- `numpy` (Matemática de matrizes)
- `pandas` (Manipulação de dados)
- `scikit-learn` (Machine Learning)
- `pillow` (Processamento de imagem)
- `scipy` (Segmentação de objetos)

```bash
pip install numpy pandas scikit-learn pillow scipy joblib
```

---

## 📖 Como Usar

### 1. Preparação
Coloque suas fotos de treinamento em:
- `data/raw/milho/`
- `data/raw/erva_daninha/`

### 2. Gerar Dataset (Processamento + Augmentation)
Este comando vai limpar as imagens, rotacioná-las em 4 ângulos e gerar o arquivo `dataset_treinamento.csv`.
```bash
python3 src/main.py
```

### 3. Treinar a Inteligência Artificial
O script lerá o CSV e criará o arquivo `modelo_plantas.pkl` com a lógica do Random Forest.
```bash
python3 ml_training/train_model.py
```

### 4. Realizar Predições
Para testar fotos novas (coloque-as na raiz de `data/raw/`):
```bash
python3 src/predict_visual.py
```
As fotos com o resultado visual (Azul para Milho e Verde para Erva-daninha) serão salvas em `data/output/`.

---

## 📊 Métricas Extraídas
O modelo decide a classe da planta baseado em:
- **Circularidade:** O quão redonda é a planta.
- **Solidez:** O quanto ela preenche sua área de ocupação.
- **Aspect Ratio:** Relação entre largura e altura (Milhos tendem a ser mais estreitos).
- **Área e Perímetro:** Tamanho e rugosidade da borda.

---

## 📧 Contato
Desenvolvido como um protótipo para agricultura de precisão.
