import os
import csv
import sys
import numpy as np

# Garante acesso aos módulos na pasta src (raiz do projeto)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, 'src'))

from core.processor import PlantProcessor
from core.extractor import FeatureExtractor

def gerar_dataset_teste():
    # Pasta de origem das fotos de teste
    raw_teste_dir = os.path.join(root_dir, "data/raw/teste")
    output_dir = os.path.join(root_dir, "data/output")
    
    # Nome do arquivo de saída (ajustado para a sequência atual)
    report_path = os.path.join(output_dir, "dataset_teste_20260422_seq02.csv")
    
    if not os.path.exists(raw_teste_dir):
        print(f"Erro: Pasta de teste nao encontrada em {raw_teste_dir}")
        return

    processor = PlantProcessor(threshold=25)
    extractor = FeatureExtractor()
    
    classes = ["milho", "erva_daninha"]
    # Para teste, geralmente nao usamos rotacao (augmentation) para ser um teste "limpo"
    # Mas se quiser, pode ativar mudando para [0, 90, 180, 270]
    rotacoes = [0] 

    print(f"\n[TEST_GEN] Gerando dataset de teste com 14 features...")
    
    total_dados = 0
    with open(report_path, mode='w', newline='') as csv_file:
        fieldnames = [
            'arquivo', 'classe', 'area_px', 'area_relativa', 'aspect_ratio', 
            'solidez', 'circularidade', 'perimetro', 'perimetro_norm',
            'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cls in classes:
            input_folder = os.path.join(raw_teste_dir, cls)
            
            if not os.path.exists(input_folder): 
                print(f"  [AVISO] Pasta da classe {cls} nao encontrada em: {input_folder}")
                continue
            
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"  Processando {len(files)} imagens de {cls}...")
            
            for filename in files:
                img_path = os.path.join(input_folder, filename)
                try:
                    img_original = processor.load_image(img_path)
                    for angulo in rotacoes:
                        img_rotated = processor.rotate_image(img_original, angulo)
                        exg = processor.get_exg(img_rotated)
                        mask = processor.create_mask(exg)
                        data = extractor.get_shape_features(mask)
                        
                        if data:
                            data['arquivo'] = filename
                            data['classe'] = cls
                            writer.writerow({k: data[k] for k in fieldnames})
                            total_dados += 1
                except Exception as e:
                    print(f"  [ERRO] {filename}: {e}")

    print(f"\n[TEST_GEN] Concluido! {total_dados} amostras salvas em: {report_path}")

if __name__ == "__main__":
    gerar_dataset_teste()
