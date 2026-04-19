import numpy as np

class FeatureExtractor:
    @staticmethod
    def get_shape_features(mask):
        """
        Extrai características geométricas de UMA única mancha verde.
        Usado para gerar o dataset de treinamento.
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        area = np.count_nonzero(mask)
        edge_h = np.diff(mask, axis=0)
        edge_v = np.diff(mask, axis=1)
        perimetro = np.count_nonzero(edge_h) + np.count_nonzero(edge_v)

        aspect_ratio = round(width / height, 4)
        solidez = round(area / (width * height), 4)
        circularidade = round((4 * np.pi * area) / (perimetro ** 2), 4) if perimetro > 0 else 0

        return {
            "area_px": area,
            "aspect_ratio": aspect_ratio,
            "solidez": solidez,
            "circularidade": circularidade,
            "perimetro": perimetro
        }
