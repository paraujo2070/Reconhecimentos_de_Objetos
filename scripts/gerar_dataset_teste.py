import os
import csv
import sys
import numpy as np

# Garante acesso aos módulos na pasta src (raiz do projeto)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, 'src'))

from core.processor import PlantProcessor
from core.extractor import FeatureExtractor
sys.path.append(os.path.join(root_dir, 'src/detection'))
from detector import FieldDetector

def gerar_dataset_teste():
    # Pasta de origem das fotos de teste
    raw_teste_dir = os.path.join(root_dir, "data/raw/teste")
    output_dir = os.path.join(root_dir, "data/output")
    
    # Nome do arquivo de saída
    report_path = os.path.join(output_dir, "dataset_teste_padrao.csv")
    
    if not os.path.exists(raw_teste_dir):
        print(f"Erro: Pasta de teste nao encontrada em {raw_teste_dir}")
        return

    processor = PlantProcessor(threshold=15)
    extractor = FeatureExtractor()
    detector = FieldDetector()
    
    classes = ["milho", "erva_daninha"]

    print(f"\n[TEST_GEN] Gerando dataset de teste (19 features)...")
    
    total_dados = 0
    with open(report_path, mode='w', newline='') as csv_file:
        fieldnames = [
            'arquivo', 'classe', 'area_relativa', 'aspect_ratio', 
            'solidez', 'circularidade', 'perimetro_norm',
            'convexidade', 'excentricidade', 'exg_medio',
            'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7',
            'zernike_1', 'zernike_2', 'zernike_3', 'zernike_4'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cls in classes:
            input_folder = os.path.join(raw_teste_dir, cls)
            if not os.path.exists(input_folder): continue
            
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"  Processando {len(files)} imagens de {cls}...")
            
            for filename in files:
                img_path = os.path.join(input_folder, filename)
                try:
                    img_original = processor.load_image(img_path)
                    exg = processor.get_exg(img_original)
                    mask = processor.create_mask(exg)
                    
                    # Filtro de 1200px (mesmo do treino)
                    plantas = detector.segment_plants(mask, min_area=1200)
                    
                    for i, planta in enumerate(plantas):
                        # Extrai o pedaço do ExG para o cálculo do ExG médio
                        exg_planta = exg[planta['bbox']]
                        data = extractor.get_shape_features(planta['mask'], exg_values=exg_planta)
                        
                        if data:
                            data['arquivo'] = f"p{i}_{filename}"
                            data['classe'] = cls
                            writer.writerow({k: data[k] for k in fieldnames})
                            total_dados += 1
                except Exception as e:
                    print(f"  [ERRO] {filename}: {e}")

    print(f"\n[TEST_GEN] Concluido! {total_dados} amostras salvas em: {report_path}")

if __name__ == "__main__":
    gerar_dataset_teste()
