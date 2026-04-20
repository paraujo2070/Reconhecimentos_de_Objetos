# AgroVision: Reconhecimento de Milho e Ervas-Daninhas 🌽🌿

Este projeto implementa um sistema de visão computacional e machine learning para identificação automatizada de plantas de milho e ervas-daninhas. Agora com suporte a **integração mobile** para coleta de dados e atualização de modelos em campo.

## 🚀 Diferenciais do Projeto
- **Sem Deep Learning:** Roda em hardware modesto (CPU) usando Random Forest.
- **Integração Mobile:** Servidor FastAPI integrado para receber fotos de celulares e enviar modelos treinados.
- **Rastreamento Fenológico:** Sistema de versionamento automático (`YYYYMMDD_seqXX`) para acompanhar o crescimento das plantas.
- **Data Augmentation:** Rotação automática para robustez do dataset.

---

## 📂 Estrutura do Projeto

```text
Reconhecimentos_de_Objetos/
├── api/
│   └── server.py           # Servidor FastAPI para comunicação com o App Android
├── data/
│   ├── raw/                # Fotos originais (coletadas via App ou Manual)
│   └── output/             # Datasets (.csv) e Modelos (.pkl) versionados
├── src/
│   ├── core/               # Processamento ExG e Extração de Métricas
│   ├── detection/          # Segmentação de plantas em fotos de campo
│   └── main.py             # Lógica de processamento (Refatorado para API)
├── ml_training/
│   └── train_model.py      # Treinamento do Random Forest (Refatorado para API)
```

---

## 🌐 API e Integração Mobile

O servidor centraliza a inteligência do projeto, permitindo que um app Android funcione como coletor e testador.

### Endpoints Principais:
1.  **`POST /upload`**: Envia fotos do campo para o servidor (Milho ou Erva).
2.  **`GET /status`**: Verifica o balanço do dataset (importante para evitar modelos tendenciosos).
3.  **`POST /train`**: Processa as novas fotos e treina uma nova versão da IA.
4.  **`GET /models`**: Lista todas as versões de inteligência disponíveis.
5.  **`GET /models/download/{file}`**: Baixa o modelo selecionado para o celular.

---

## 📖 Como Usar

### 1. Preparação e Instalação
```bash
pip install numpy pandas scikit-learn pillow scipy joblib fastapi uvicorn python-multipart
```

### 2. Iniciando o Servidor (Modo API)
Para permitir que o app Android conecte ao seu PC:
```bash
python3 api/server.py
```
*Acesse `http://localhost:8000/docs` para ver a documentação interativa.*

### 3. Treinamento Manual (Opcional)
Se preferir rodar localmente sem a API:
```bash
python3 src/main.py           # Gera o CSV
python3 ml_training/train_model.py  # Gera o PKL
```

---

## 📏 Nova Lógica de Versionamento
Para acompanhar plantas que saíram da terra recentemente (ex: 5 dias), o sistema gera arquivos com carimbo de data:
- **Dataset:** `data/output/dataset_20260420_seq01.csv`
- **Modelo:** `data/output/modelo_20260420_seq01.pkl`

Isso permite que você mantenha uma biblioteca de modelos para cada estágio de crescimento da cultura.

---

## 📧 Contato
Desenvolvido para agricultura de precisão e monitoramento inteligente de safras.
