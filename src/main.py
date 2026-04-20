import os
import csv
import sys

# Garante que os módulos locais sejam encontrados
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from processor import PlantProcessor
from extractor import FeatureExtractor

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "data/raw")
    output_dir = os.path.join(root_dir, "data/output")
    
    os.makedirs(output_dir, exist_ok=True)

    processor = PlantProcessor(threshold=25)
    extractor = FeatureExtractor()
    
    report_path = os.path.join(output_dir, "dataset_treinamento.csv")
    classes = ["milho", "erva_daninha"]
    rotacoes = [0, 90, 180, 270] # Data Augmentation

    print(f"{'='*60}")
    print(f"GERADOR DE DATASET COM DATA AUGMENTATION (ROTAÇÃO)")
    print(f"{'='*60}\n")

    total_fotos = 0
    with open(report_path, mode='w', newline='') as csv_file:
        fieldnames = ['arquivo', 'classe', 'area_px', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cls in classes:
            input_folder = os.path.join(raw_dir, cls)
            if not os.path.exists(input_folder): continue
            
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
            total_fotos += len(files)
            print(f"Classe: {cls.upper()} | {len(files)} fotos originais.")

            for filename in files:
                img_path = os.path.join(input_folder, filename)
                
                try:
                    img_original = processor.load_image(img_path)
                    
                    for angulo in rotacoes:
                        # 1. Rotaciona a imagem
                        img_rotated = processor.rotate_image(img_original, angulo)
                        
                        # 2. Processa (Máscara)
                        exg = processor.get_exg(img_rotated)
                        mask = processor.create_mask(exg)
                        
                        # 3. Extrai características
                        data = extractor.get_shape_features(mask)
                        
                        if data:
                            data['arquivo'] = f"{angulo}_{filename}"
                            data['classe'] = cls
                            writer.writerow({k: data[k] for k in fieldnames})
                    
                    print(f"  [OK] {filename} (Geradas 4 variações)")
                        
                except Exception as e:
                    print(f"  [ERRO] {filename}: {e}")

    print(f"\n{'='*60}")
    print(f"DATASET AMPLIADO COM SUCESSO!")
    print(f"Total de linhas (dados): {total_fotos * len(rotacoes)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
