import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def run_training(csv_path=None, model_output_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_csv = csv_path if csv_path else os.path.join(base_dir, "data/output/dataset_treinamento.csv")
    target_model = model_output_path if model_output_path else os.path.join(base_dir, "data/output/modelo_plantas.pkl")
    
    if not os.path.exists(target_csv):
        print(f"[TRAINER] Erro: Arquivo {target_csv} não encontrado.")
        return None

    print(f"[TRAINER] Lendo dataset: {target_csv}")
    df = pd.read_csv(target_csv)
    
    features = ['area_px', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro']
    X = df[features]
    y = df['classe']

    # Treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Avaliação simples para o log
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[TRAINER] Modelo treinado com {accuracy*100:.2f}% de acurácia.")

    joblib.dump(model, target_model)
    print(f"[TRAINER] Salvo em: {target_model}")
    return target_model

if __name__ == "__main__":
    run_training()
