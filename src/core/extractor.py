import numpy as np
import cv2
import mahotas

class FeatureExtractor:
    @staticmethod
    def get_shape_features(mask, exg_values=None):
        """
        Extrai 19 características. 
        Suporta máscaras com múltiplos objetos (Análise Global).
        """
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

        mask_cv = mask.astype(np.uint8)
        area = np.count_nonzero(mask)
        
        # Bounding Box Global
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Encontrar todos os contornos (plantas/fragmentos)
        contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        # Unir todos os pontos de todos os contornos para cálculos globais
        all_points = np.vstack(contours)
        
        # Perímetro Total (Soma de todos os fragmentos)
        perimetro = sum([cv2.arcLength(cnt, True) for cnt in contours])
        
        # Perímetro do Casco Convexo Global (Envolve toda a vegetação)
        hull = cv2.convexHull(all_points)
        perimetro_convexo = cv2.arcLength(hull, True)
        
        # 1. Convexidade Global
        convexidade = round(perimetro_convexo / perimetro, 4) if perimetro > 0 else 0
        
        # 2. Excentricidade Global (Elipse que envolve tudo)
        excentricidade = 0
        if len(all_points) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(all_points)
                a = ma / 2
                b = MA / 2
                if a > 0:
                    excentricidade = round(np.sqrt(1 - (min(a, b)**2 / max(a, b)**2)), 4)
            except: excentricidade = 0

        # 3. ExG Médio (De todo o verde)
        exg_medio = 0
        if exg_values is not None:
            exg_medio = round(np.mean(exg_values[mask > 0]), 4)

        # Características base
        area_total = mask.shape[0] * mask.shape[1]
        area_relativa = round(area / area_total, 6)
        perimetro_norm = round(perimetro / (area ** 0.5), 4) if area > 0 else 0
        aspect_ratio = round(width / height, 4)
        solidez = round(area / (width * height), 4)
        circularidade = round((4 * np.pi * area) / (perimetro ** 2), 4) if perimetro > 0 else 0

        # Momentos de Hu (Nativamente globais)
        moments = cv2.moments(mask_cv)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_log = []
        for i in range(0, 7):
            val = hu_moments[i]
            if val != 0:
                hu_log.append(round(-1 * np.sign(val) * np.log10(abs(val)), 6))
            else:
                hu_log.append(0.0)

        # 4. Momentos de Zernike (Nativamente globais)
        radius = max(width, height) / 2
        z_moments = mahotas.features.zernike_moments(mask_cv, radius, degree=4)
        z_feats = [round(float(z), 6) for z in z_moments[:4]]

        return {
            "area_relativa": area_relativa,
            "aspect_ratio": aspect_ratio,
            "solidez": solidez,
            "circularidade": circularidade,
            "perimetro_norm": perimetro_norm,
            "convexidade": convexidade,
            "excentricidade": excentricidade,
            "exg_medio": exg_medio,
            "hu_1": hu_log[0], "hu_2": hu_log[1], "hu_3": hu_log[2],
            "hu_4": hu_log[3], "hu_5": hu_log[4], "hu_6": hu_log[5], "hu_7": hu_log[6],
            "zernike_1": z_feats[0], "zernike_2": z_feats[1], 
            "zernike_3": z_feats[2], "zernike_4": z_feats[3]
        }
