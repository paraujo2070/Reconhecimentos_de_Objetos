import os
import sys
import joblib
import numpy as np
from PIL import Image, ImageDraw

# Sobe um nível para achar a raiz do projeto e a pasta src
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, 'src'))

from core.processor import PlantProcessor
from core.extractor import FeatureExtractor

def main():
    # 1. Configuração de Caminhos
    model_path = os.path.join(root_dir, "data/output/modelo_20260422_seq02.pkl")
    raw_dir = os.path.join(root_dir, "data/raw")
    output_dir = os.path.join(root_dir, "data/output/predicoes_diretas")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. Carregar o Modelo Treinado e Componentes
    if not os.path.exists(model_path):
        print("Erro: Modelo não encontrado em data/output/modelo_plantas.pkl")
        return
    
    model = joblib.load(model_path)
    processor = PlantProcessor(threshold=15)
    extractor = FeatureExtractor()

    # 3. Localizar fotos para teste (na raiz de data/raw)
    target_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(raw_dir, f))]
    
    if not target_files:
        print(f"Nenhuma foto encontrada para teste direto em {raw_dir}")
        return

    print(f"{'='*60}")
    print(f"MODO DE INFERÊNCIA DIRETA (UMA CHAMADA POR FOTO)")
    print(f"{'='*60}\n")

    for filename in target_files:
        img_path = os.path.join(raw_dir, filename)
        print(f"Analisando: {filename}")

        # --- ETAPA 1: PROCESSAMENTO ---
        img_pil = Image.open(img_path).convert("RGB")
        img_array = np.array(img_pil)
        
        exg = processor.get_exg(img_array)
        mask = processor.create_mask(exg)

        # --- ETAPA 2: EXTRAÇÃO DE CARACTERÍSTICAS ---
        # Analisa a imagem como um todo (foco na planta principal)
        features = extractor.get_shape_features(mask)
        
        if features:
            # Organiza os dados para o modelo (7 features agora)
            input_data = [[
                features['area_px'], 
                features['area_relativa'],
                features['aspect_ratio'], 
                features['solidez'], 
                features['circularidade'], 
                features['perimetro'],
                features['perimetro_norm']
            ]]

            # --- ETAPA 3: PREDIÇÃO ÚNICA ---
            prediction = model.predict(input_data)[0]
            probabilidades = model.predict_proba(input_data)[0]
            confianca = np.max(probabilidades) * 100

            # --- ETAPA 4: RESULTADO VISUAL ---
            draw = ImageDraw.Draw(img_pil)
            
            # Define cor baseada na predição: Milho (Azul), Erva (Verde)
            color = (0, 0, 255) if prediction == "milho" else (0, 255, 0)
            text_label = f"RESULTADO: {prediction.upper()} ({confianca:.1f}%)"

            # Desenha uma borda externa na imagem para indicar a classificação
            draw.rectangle([0, 0, img_pil.size[0]-1, img_pil.size[1]-1], outline=color, width=20)
            
            # Adiciona o texto informativo
            print(f"  -> {text_label}")

            # Salvar o resultado
            save_path = os.path.join(output_dir, f"resultado_{filename}")
            img_pil.save(save_path)
            print(f"  -> Salvo em: {save_path}\n")
        else:
            print(f"  -> [AVISO] Nenhuma vegetação detectada em {filename}\n")

if __name__ == "__main__":
    main()
