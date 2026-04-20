import numpy as np
from PIL import Image
import os

class PlantProcessor:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def load_image(self, image_path):
        """Carrega imagem e converte para array NumPy."""
        img = Image.open(image_path)
        return np.array(img)

    def get_exg(self, img_array):
        """
        Calcula o Excess Green Index (ExG): 2G - R - B
        Isola o que é verde na imagem.
        """
        # Convertendo para float para evitar overflow no uint8
        r = img_array[:, :, 0].astype(float)
        g = img_array[:, :, 1].astype(float)
        b = img_array[:, :, 2].astype(float)
        
        exg = 2*g - r - b
        return exg

    def create_mask(self, exg_array):
        """Cria uma máscara binária baseada no threshold."""
        mask = np.where(exg_array > self.threshold, 255, 0).astype(np.uint8)
        return mask

    def apply_mask(self, img_array, mask):
        """Aplica a máscara na imagem original para ver apenas o que é verde."""
        # Expande a máscara de (H, W) para (H, W, 3) para bater com as cores
        mask_3d = np.stack([mask]*3, axis=-1) / 255
        result = (img_array * mask_3d).astype(np.uint8)
        return result

    def rotate_image(self, img_array, angle):
        """Rotaciona a imagem em 0, 90, 180 ou 270 graus."""
        if angle == 90:
            return np.rot90(img_array, k=1)
        elif angle == 180:
            return np.rot90(img_array, k=2)
        elif angle == 270:
            return np.rot90(img_array, k=3)
        return img_array

    def process_and_save(self, input_path, output_folder):
        """Executa todo o fluxo e salva o resultado."""
        filename = os.path.basename(input_path)
        img_array = self.load_image(input_path)
        
        # 1. Isolar verde
        exg = self.get_exg(img_array)
        mask = self.create_mask(exg)
        result = self.apply_mask(img_array, mask)
        
        # Salva o resultado
        output_path = os.path.join(output_folder, f"processed_{filename}")
        Image.fromarray(result).save(output_path)
        return output_path
