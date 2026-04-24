import numpy as np
import cv2

class FeatureExtractor:
    @staticmethod
    def get_shape_features(mask):
        """
        Extrai características geométricas de UMA única mancha verde.
        Inclui Momentos de Hu para invariância de escala e rotação.
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

        # Garantir que a máscara seja uint8 para o OpenCV
        mask_cv = mask.astype(np.uint8)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        area = np.count_nonzero(mask)
        edge_h = np.diff(mask_cv, axis=0)
        edge_v = np.diff(mask_cv, axis=1)
        perimetro = np.count_nonzero(edge_h) + np.count_nonzero(edge_v)

        # Features de invariância de escala
        area_total = mask.shape[0] * mask.shape[1]
        area_relativa = round(area / area_total, 6)
        perimetro_norm = round(perimetro / (area ** 0.5), 4) if area > 0 else 0

        aspect_ratio = round(width / height, 4)
        solidez = round(area / (width * height), 4)
        circularidade = round((4 * np.pi * area) / (perimetro ** 2), 4) if perimetro > 0 else 0

        # Cálculo dos Momentos de Hu (7 features invariantes)
        moments = cv2.moments(mask_cv)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Logaritmo para normalizar a escala dos momentos (opcional mas recomendado)
        # Evita números extremamente pequenos que podem causar erro numérico
        hu_log = []
        for i in range(0, 7):
            val = hu_moments[i]
            if val != 0:
                hu_log.append(round(-1 * np.sign(val) * np.log10(abs(val)), 6))
            else:
                hu_log.append(0.0)

        return {
            "area_relativa": area_relativa,
            "aspect_ratio": aspect_ratio,
            "solidez": solidez,
            "circularidade": circularidade,
            "perimetro_norm": perimetro_norm,
            "hu_1": hu_log[0],
            "hu_2": hu_log[1],
            "hu_3": hu_log[2],
            "hu_4": hu_log[3],
            "hu_5": hu_log[4],
            "hu_6": hu_log[5],
            "hu_7": hu_log[6]
        }
