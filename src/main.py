import os
import csv
import sys

# Garante que os módulos locais sejam encontrados
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from processor import PlantProcessor
from extractor import FeatureExtractor

def run_analysis_pipeline(image_path, processor, extractor, output_img_dir):
    """
    Executa o pipeline sequencial:
    1. Executa o Processor (Limpeza e Segmentação)
    2. Executa o Extractor (Cálculo de métricas geométricas)
    """
    # ETAPA 1: PROCESSAMENTO (Lógica do processor.py)
    img_array = processor.load_image(image_path)
    exg = processor.get_exg(img_array)
    mask = processor.create_mask(exg)
    
    # Salva o resultado visual (Auditoria Comercial)
    processor.process_and_save(image_path, output_img_dir)
    
    # ETAPA 2: EXTRAÇÃO (Lógica do extractor.py)
    features = extractor.get_shape_features(mask)
    
    return features

def main():
    # Setup de diretórios
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "data/raw")
    processed_dir = os.path.join(root_dir, "data/processed")
    output_dir = os.path.join(root_dir, "data/output")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Instanciação das classes
    processor = PlantProcessor(threshold=25)
    extractor = FeatureExtractor()
    
    report_path = os.path.join(output_dir, "dataset_treinamento.csv")
    classes = ["milho", "erva_daninha"]

    print(f"{'='*60}")
    print(f"PIPELINE SEQUENCIAL: PROCESSAMENTO -> EXTRAÇÃO")
    print(f"{'='*60}\n")

    with open(report_path, mode='w', newline='') as csv_file:
        fieldnames = ['arquivo', 'classe', 'area_px', 'aspect_ratio', 'solidez', 'circularidade', 'perimetro']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cls in classes:
            input_folder = os.path.join(raw_dir, cls)
            if not os.path.exists(input_folder): continue
            
            output_img_cls = os.path.join(processed_dir, cls)
            os.makedirs(output_img_cls, exist_ok=True)

            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
            print(f"Classe: {cls.upper()} | {len(files)} imagens encontradas.")

            for filename in files:
                img_path = os.path.join(input_folder, filename)
                
                try:
                    # EXECUÇÃO DO PIPELINE
                    data = run_analysis_pipeline(img_path, processor, extractor, output_img_cls)
                    
                    if data:
                        data['arquivo'] = filename
                        data['classe'] = cls
                        # Gravação seletiva das colunas do CSV
                        writer.writerow({k: data[k] for k in fieldnames})
                        print(f"  [OK] {filename} processado e analisado.")
                        
                except Exception as e:
                    print(f"  [ERRO] Falha ao processar {filename}: {e}")

    print(f"\n{'='*60}")
    print(f"PROCESSO FINALIZADO!")
    print(f"Relatório: {report_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
