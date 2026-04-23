import joblib
import os
import pandas as pd
import numpy as np

def ver_importancia():
    # Detecta a raiz do projeto (um nível acima da pasta scripts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Apontando para o novo modelo seq02
    model_path = os.path.join(base_dir, 'data/output/modelo_20260423_seq02.pkl')
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo {model_path} nao encontrado.")
        return

    try:
        # Carrega o pipeline
        pipeline = joblib.load(model_path)
        
        # O classificador esta no passo 'rf' (index 1 do pipeline)
        rf_model = pipeline.steps[1][1]
        
        # Lista atualizada com as 14 features
        features = [
            'area_px', 'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro', 'perimetro_norm',
            'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7'
        ]
        
        # Pega as importancias
        importancias = rf_model.feature_importances_
        
        # Cria um DataFrame para visualizacao
        df_imp = pd.DataFrame({
            'Feature': features,
            'Importancia (%)': importancias * 100
        })
        
        # Ordena da mais importante para a menos importante
        df_imp = df_imp.sort_values(by='Importancia (%)', ascending=False)

        print("\n" + "="*50)
        print("IMPORTANCIA DAS CARACTERISTICAS - modelo_20260423_seq02 (14 FEATURES)")
        print("="*50)
        print(df_imp.to_string(index=False, float_format=lambda x: "{:.2f}%".format(x)))
        print("="*50)
        
        # Analise de dominancia
        hu_total = df_imp[df_imp['Feature'].str.contains('hu')]['Importancia (%)'].sum()
        area_total = df_imp[df_imp['Feature'].isin(['area_px', 'perimetro'])]['Importancia (%)'].sum()
        
        print(f"\nRESUMO DE INFLUENCIA:")
        print(f"- Importancia Total dos Momentos de Hu: {hu_total:.2f}%")
        print(f"- Importancia do Tamanho Absoluto (Area/Perimetro): {area_total:.2f}%")
        
        if hu_total > area_total:
            print("\nSUCESSO: O modelo agora valoriza mais a FORMA (Hu) do que o TAMANHO.")
        else:
            print("\nAVISO: O tamanho absoluto ainda tem peso relevante. Considere variar mais as distancias no treino.")

    except Exception as e:
        print(f"Erro ao analisar o modelo: {e}")

if __name__ == "__main__":
    ver_importancia()
