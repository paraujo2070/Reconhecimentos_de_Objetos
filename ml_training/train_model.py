import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def run_training(csv_path=None, model_output_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_csv = csv_path if csv_path else os.path.join(base_dir, "data/output/dataset_treinamento.csv")
    target_model = model_output_path if model_output_path else os.path.join(base_dir, "data/output/modelo_plantas.pkl")
    
    # Define o caminho do ONNX baseado no nome do modelo
    onnx_output_path = target_model.replace(".pkl", ".onnx")
    
    if not os.path.exists(target_csv):
        print(f"[TRAINER] Erro: Arquivo {target_csv} não encontrado.")
        return None

    print(f"[TRAINER] Lendo dataset: {target_csv}")
    df = pd.read_csv(target_csv)
    
    # Features atualizadas (14 no total)
    features = [
        'area_px', 'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro', 'perimetro_norm',
        'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7'
    ]
    X = df[features].values.astype(np.float32)
    y = df['classe']
    
    # Criar grupos baseados no nome do arquivo ORIGINAL
    # Formato esperado: "angulo_p0_nomeoriginal.jpg" -> Extraímos apenas "nomeoriginal.jpg"
    def extrair_nome_original(nome):
        partes = nome.split('_')
        # Se o nome tem o padrão angulo_pX_nome, o nome original começa na posição 2
        if len(partes) > 2 and partes[1].startswith('p'):
            return "_".join(partes[2:])
        return nome

    groups = df['arquivo'].apply(extrair_nome_original)

    # Separação baseada em grupos
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Criar Pipeline (Normalização + Classificador Robusto)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            min_samples_leaf=2, 
            random_state=42, 
            class_weight='balanced'
        ))
    ])
    
    pipeline.fit(X_train, y_train)

    # Avaliação
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[TRAINER] Modelo treinado com {accuracy*100:.2f}% de acurácia (Validação por Grupos).")
    print(classification_report(y_test, y_pred))

    # 1. Salva PKL (Pipeline completo)
    joblib.dump(pipeline, target_model)
    
    # 2. Salva ONNX (Pipeline completo para Android)
    try:
        # Agora temos 14 features
        initial_type = [('float_input', FloatTensorType([None, 14]))]
        onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
        with open(onnx_output_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"[TRAINER] Exportado para ONNX (com Scaler): {onnx_output_path}")
    except Exception as e:
        print(f"[TRAINER] Erro ao exportar ONNX: {e}")

    return target_model

if __name__ == "__main__":
    import sys
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_training(csv_path=csv_arg, model_output_path=model_arg)
