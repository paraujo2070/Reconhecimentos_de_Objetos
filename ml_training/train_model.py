import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # 1. Carregar o Dataset gerado pelo outro módulo
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data/output/dataset_treinamento.csv")
    
    if not os.path.exists(csv_path):
        print(f"Erro: O arquivo {csv_path} não foi encontrado. Execute o processamento primeiro!")
        return

    df = pd.read_csv(csv_path)
    
    # 2. Selecionar as características (Features) e o Alvo (Target)
    # Não usamos 'arquivo' porque é apenas um nome
    features = ['area_px', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro']
    X = df[features]
    y = df['classe']

    # 3. Dividir dados em Treino e Teste (80% treino, 20% teste)
    # Com 79 amostras, o teste terá cerca de 16 imagens.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Treinando com {len(X_train)} amostras e testando com {len(X_test)}...")

    # 4. Instanciar e Treinar o Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Avaliar o Modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"RESULTADO DO TREINAMENTO")
    print(f"{'='*40}")
    print(f"Acurácia: {accuracy * 100:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # 6. Salvar o modelo treinado para uso futuro
    model_output = os.path.join(base_dir, "data/output/modelo_plantas.pkl")
    joblib.dump(model, model_output)
    
    print(f"\nModelo salvo em: {model_output}")
    
    # 7. Mostrar quais métricas foram mais importantes
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nImportância das métricas para a decisão:")
    print(importances)

if __name__ == "__main__":
    main()
