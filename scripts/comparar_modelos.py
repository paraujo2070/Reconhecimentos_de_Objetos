import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def validar():
    # Sobe um nível para achar a raiz do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data/output")
    
    # 1. Carregar o Dataset de Teste de HOJE (Quarentena)
    test_csv = os.path.join(output_dir, "dataset_teste_20260423_seq03.csv")
    if not os.path.exists(test_csv):
        # Tenta a seq03 se a 02 não existir
        test_csv = os.path.join(output_dir, "dataset_teste_20260421_seq02.csv")
    
    if not os.path.exists(test_csv):
        print(f"Erro: Dataset de teste não encontrado!")
        return

    df_test = pd.read_csv(test_csv)
    y_true = df_test['classe']
    classes_nomes = sorted(y_true.unique())

    # Preparar as versões de X (3, 5 e 12 features)
    features_3 = ['aspect_ratio', 'solidez', 'circularidade']
    features_5 = ['area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro_norm']
    features_12 = [
        'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro_norm',
        'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7'
    ]
    
    X_dict = {
        3: df_test[features_3].values.astype(np.float32) if all(c in df_test.columns for c in features_3) else None,
        5: df_test[features_5].values.astype(np.float32) if all(c in df_test.columns for c in features_5) else None,
        12: df_test[features_12].values.astype(np.float32) if all(c in df_test.columns for c in features_12) else None,
        14: None # Desativado
    }

    def testar_modelo(caminho, label):
        if not os.path.exists(caminho):
            print(f"\n[AVISO] {label} não encontrado em: {os.path.basename(caminho)}")
            return
        
        try:
            model = joblib.load(caminho)
            
            # Detectar n_features necessário
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif hasattr(model, 'steps'): # Se for Pipeline (nosso caso)
                # O Scaler é o primeiro step e ele sabe quantas features recebeu
                n_features = model.steps[0][1].n_features_in_
            else:
                n_features = 12 if "seq" in caminho else 5

            X_input = X_dict.get(n_features)
            
            if X_input is None:
                print(f"\n[ERRO] O modelo {os.path.basename(caminho)} exige {n_features} features, mas o CSV de teste nao as possui.")
                return
            
            y_pred = model.predict(X_input)
            acc = accuracy_score(y_true, y_pred)
            
            print(f"\n" + "-"*30)
            print(f"RESULTADO: {label}")
            print(f"Arquivo: {os.path.basename(caminho)}")
            print(f"Features detectadas: {n_features}")
            print(f"Acurácia: {acc*100:.2f}%")
            
            if n_features == 7: # Mostrar detalhes apenas para o modelo novo/principal
                print("\nRelatório de Classificação:")
                print(classification_report(y_true, y_pred))
                print("Matriz de Confusão:")
                cm = confusion_matrix(y_true, y_pred)
                print(pd.DataFrame(cm, index=[f"Real {c}" for c in classes_nomes], 
                                     columns=[f"Predito {c}" for c in classes_nomes]))
        
        except Exception as e:
            print(f"\n[ERRO EM {label}]: {e}")

    print(f"\n" + "="*50)
    print(f" COMPARATIVO DE MODELOS - TESTE EM {len(df_test)} AMOSTRAS")
    print("="*50)

    # Testar o modelo de Ontem (Original)
    testar_modelo(os.path.join(output_dir, "modelo_20260423_seq01.pkl"), "MODELO ANTERIOR (HU)")
    
    # Testar o modelo de Hoje (Último gerado)
    testar_modelo(os.path.join(output_dir, "modelo_20260423_seq02.pkl"), "MODELO ATUAL (HU + SEG)")


if __name__ == "__main__":
    validar()
