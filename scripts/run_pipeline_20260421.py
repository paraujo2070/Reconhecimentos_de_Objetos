import os
import sys
from datetime import datetime

# Sobe um nível para achar a raiz do projeto
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(os.path.join(root_dir, 'ml_training'))

from main import run_processing
from train_model import run_training

def main():
    date_str = datetime.now().strftime("%Y%m%d")
    seq = "seq02" # Nova rodada com segmentação no treino
    
    output_dir = os.path.join(root_dir, "data/output")
    
    # 1. Gerar Dataset de Treinamento (Fotos de Ontem/Base)
    train_csv = os.path.join(output_dir, f"dataset_treino_{date_str}_{seq}.csv")
    print(f"\n--- GERANDO DATASET DE TREINO ---")
    run_processing(output_csv_path=train_csv)
    
    # 2. Gerar Dataset de Teste (Fotos de HOJE)
    test_csv = os.path.join(output_dir, f"dataset_teste_{date_str}_{seq}.csv")
    print(f"\n--- GERANDO DATASET DE TESTE (QUARENTENA) ---")
    # Ao passar um caminho com "teste", o src/main.py buscará em data/raw/teste/
    run_processing(output_csv_path=test_csv)
    
    # 3. Treinar Modelo
    model_pkl = os.path.join(output_dir, f"modelo_{date_str}_{seq}.pkl")
    print(f"\n--- TREINANDO MODELO ---")
    run_training(csv_path=train_csv, model_output_path=model_pkl)

if __name__ == "__main__":
    main()
