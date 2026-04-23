import os
import sys
import shutil
import numpy as np
from PIL import Image

# Adiciona src ao path para usar os módulos do projeto
root_dir = os.getcwd()
sys.path.append(os.path.join(root_dir, 'src'))

from core.processor import PlantProcessor
from detection.detector import FieldDetector

def filtrar_por_densidade(limite_fragmentos=30):
    raw_dir = os.path.join(root_dir, "data/raw")
    quarentena_dir = os.path.join(root_dir, "data/raw_alta_densidade")
    
    classes = ["milho", "erva_daninha"]
    processor = PlantProcessor(threshold=25)
    detector = FieldDetector()

    print(f"\n[FILTRO] Iniciando limpeza por densidade (Limite: {limite_fragmentos} fragmentos)...")

    for cls in classes:
        input_folder = os.path.join(raw_dir, cls)
        target_folder = os.path.join(quarentena_dir, cls)
        
        if not os.path.exists(input_folder):
            continue
            
        os.makedirs(target_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        removidos = 0
        print(f"  Analisando {len(files)} imagens de {cls}...")

        for filename in files:
            img_path = os.path.join(input_folder, filename)
            try:
                img_array = processor.load_image(img_path)
                exg = processor.get_exg(img_array)
                mask = processor.create_mask(exg)
                
                # Conta quantos fragmentos existem na foto original
                plantas = detector.segment_plants(mask, min_area=500)
                num_fragmentos = len(plantas)
                
                if num_fragmentos > limite_fragmentos:
                    # Move para a pasta de alta densidade
                    shutil.move(img_path, os.path.join(target_folder, filename))
                    removidos += 1
            except Exception as e:
                print(f"    [ERRO] {filename}: {e}")

        print(f"  -> {cls}: {removidos} imagens movidas para quarentena.")

    print("\n[FILTRO] Limpeza concluída.")

if __name__ == "__main__":
    # Podemos ajustar o limite aqui. 30 é um bom começo para evitar ruído.
    filtrar_por_densidade(limite_fragmentos=30)
