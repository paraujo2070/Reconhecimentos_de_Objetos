import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def validar():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data/output")
    
    # 1. Carregar o Dataset de Teste de HOJE (Quarentena)
    test_csv = os.path.join(output_dir, "dataset_teste_20260421_seq02.csv")
    if not os.path.exists(test_csv):
        # Tenta a seq03 se a 02 não existir
        test_csv = os.path.join(output_dir, "dataset_teste_20260421_seq03.csv")
    
    if not os.path.exists(test_csv):
        print(f"Erro: Dataset de teste não encontrado!")
        return

    df_test = pd.read_csv(test_csv)
    y_true = df_test['classe']
    classes_nomes = sorted(y_true.unique())

    # Preparar as duas versões de X
    features_5 = ['area_px', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro']
    features_7 = ['area_px', 'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro', 'perimetro_norm']
    
    X_5 = df_test[features_5].values.astype(np.float32)
    X_7 = df_test[features_7].values.astype(np.float32)

    def testar_modelo(caminho, label):
        if not os.path.exists(caminho):
            print(f"\n[AVISO] {label} não encontrado em: {os.path.basename(caminho)}")
            return
        
        try:
            model = joblib.load(caminho)
            
            # Detectar automaticamente se o modelo quer 5 ou 7 features
            # Algumas versões do sklearn usam n_features_in_
            try:
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                elif hasattr(model, 'steps'): # Se for Pipeline
                    n_features = model.steps[0][1].n_features_in_
                else:
                    n_features = 5 # Fallback
            except:
                n_features = 7 if "20260421" in caminho or "seq" in caminho else 5

            X_input = X_7 if n_features == 7 else X_5
            
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
    testar_modelo(os.path.join(output_dir, "modelo_20260421_seq03.pkl"), "MODELO ANTERIOR")
    
    # Testar o modelo de Hoje (Último gerado)
    testar_modelo(os.path.join(output_dir, "modelo_20260422_seq01.pkl".replace(".onnx", ".pkl")), "MODELO ATUAL")

if __name__ == "__main__":
    validar()
