import os
import csv
import sys

# Garante que os módulos locais sejam encontrados
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from processor import PlantProcessor
from extractor import FeatureExtractor

def run_processing(output_csv_path=None):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "data/raw")
    output_dir = os.path.join(root_dir, "data/output")
    
    os.makedirs(output_dir, exist_ok=True)

    processor = PlantProcessor(threshold=25)
    extractor = FeatureExtractor()
    
    report_path = output_csv_path if output_csv_path else os.path.join(output_dir, "dataset_treinamento.csv")
    classes = ["milho", "erva_daninha"]
    rotacoes = [0, 90, 180, 270] # Data Augmentation

    print(f"\n[PROCESSOR] Iniciando extração de características...")
    
    total_dados = 0
    with open(report_path, mode='w', newline='') as csv_file:
        fieldnames = ['arquivo', 'classe', 'area_px', 'area_relativa', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro', 'perimetro_norm']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cls in classes:
            # Se output_csv_path contiver "teste", busca na pasta raw/teste/classe
            # Caso contrário, busca na pasta raw/classe
            sub_folder = "teste" if output_csv_path and "teste" in output_csv_path else ""
            input_folder = os.path.join(raw_dir, sub_folder, cls)
            
            if not os.path.exists(input_folder): 
                print(f"  [AVISO] Pasta não encontrada: {input_folder}")
                continue
            
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
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
                            data['arquivo'] = f"{angulo}_{filename}"
                            data['classe'] = cls
                            writer.writerow({k: data[k] for k in fieldnames})
                            total_dados += 1
                except Exception as e:
                    print(f"  [ERRO] {filename}: {e}")

    print(f"[PROCESSOR] Concluído. {total_dados} amostras geradas em {report_path}")
    return report_path, total_dados

if __name__ == "__main__":
    run_processing()
