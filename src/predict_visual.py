import os
import sys
import joblib
import numpy as np
from PIL import Image, ImageDraw

# Caminhos para os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'detection'))

from processor import PlantProcessor
from extractor import FeatureExtractor
from detector import FieldDetector

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "data/output/modelo_plantas.pkl")
    raw_dir = os.path.join(root_dir, "data/raw")
    output_dir = os.path.join(root_dir, "data/output/predicoes_campo")
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print("Erro: Modelo não encontrado.")
        return
    
    model = joblib.load(model_path)
    processor = PlantProcessor()
    extractor = FeatureExtractor()
    detector = FieldDetector()

    target_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(raw_dir, f))]

    print(f"{'='*60}")
    print(f"SISTEMA DE DETECÇÃO DE CAMPO")
    print(f"{'='*60}\n")

    for filename in target_files:
        img_path = os.path.join(raw_dir, filename)
        img_pil = Image.open(img_path).convert("RGB")
        img_array = np.array(img_pil)
        
        # 1. Processamento (Core)
        exg = processor.get_exg(img_array)
        mask = processor.create_mask(exg)

        # 2. Detecção (Detection) - Segmenta a imagem em várias plantas
        detected_plants = detector.segment_plants(mask)
        
        draw = ImageDraw.Draw(img_pil)
        count_milho, count_erva = 0, 0

        for plant in detected_plants:
            # 3. Extração (Core) - Extrai métricas de cada planta segmentada
            features = extractor.get_shape_features(plant['mask'])
            
            if features:
                input_data = [[
                    features['area_px'], features['aspect_ratio'], 
                    features['solidez'], features['circularidade'], 
                    features['perimetro']
                ]]

                # 4. Predição (ML)
                prediction = model.predict(input_data)[0]
                
                # Coordenadas
                slice_y, slice_x = plant['bbox']
                y_min, y_max, x_min, x_max = slice_y.start, slice_y.stop, slice_x.start, slice_x.stop

                color = (0, 0, 255) if prediction == "milho" else (0, 255, 0)
                if prediction == "milho": count_milho += 1
                else: count_erva += 1

                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)

        save_path = os.path.join(output_dir, f"analise_{filename}")
        img_pil.save(save_path)
        print(f"[PROCESSADO] {filename}: Milhos: {count_milho} | Ervas: {count_erva}")

if __name__ == "__main__":
    main()
