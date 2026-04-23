import pandas as pd
import numpy as np
import os
import sys

# Tenta importar joblib ou pickle de forma segura
try:
    import joblib
    def load_model(path):
        return joblib.load(path)
except ImportError:
    import pickle
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def analisar():
    # Detecta a raiz do projeto (um nível acima da pasta scripts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'data/output/modelo_20260422_seq01.pkl')
    test_csv = os.path.join(base_dir, 'data/output/dataset_teste_20260421_seq02.csv')
    
    if not os.path.exists(test_csv):
        test_csv = os.path.join(base_dir, 'data/output/dataset_teste_20260421_seq03.csv')

    if not os.path.exists(model_path) or not os.path.exists(test_csv):
        print(f"Erro: Arquivos nao encontrados ({model_path} ou {test_csv})")
        return

    try:
        model = load_model(model_path)
        df_test = pd.read_csv(test_csv)

        features = ['area_px', 'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro', 'perimetro_norm']
        X = df_test[features].values.astype(np.float32)
        y_true = df_test['classe']
        y_pred = model.predict(X)

        erros = df_test[(y_true == 'milho') & (y_pred == 'erva_daninha')]

        print("\n" + "="*80)
        print("DETALHE DOS 4 ERROS (MILHO -> ERVA_DANINHA)")
        print("="*80)
        
        if not erros.empty:
            cols_analise = ['arquivo', 'solidez', 'circularidade', 'aspect_ratio', 'perimetro_norm']
            print(erros[cols_analise].to_string(index=False))
            
            print("\nMedias do Milho Correto (para comparacao):")
            corretos = df_test[(y_true == 'milho') & (y_pred == 'milho')]
            print(corretos[cols_analise[1:]].mean().to_frame().T.to_string(index=False))
        else:
            print("Nenhum erro encontrado.")
    except Exception as e:
        print(f"Erro durante a analise: {e}")

if __name__ == "__main__":
    analisar()
