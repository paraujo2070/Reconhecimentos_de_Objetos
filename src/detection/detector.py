import numpy as np
from scipy.ndimage import label, find_objects

class FieldDetector:
    @staticmethod
    def segment_plants(mask, min_area=300):
        """
        Encontra e isola cada planta individual em uma foto ampla.
        Retorna uma lista de tuplas (mascara_isolada, coordenadas).
        """
        labeled_mask, num_features = label(mask)
        objects = find_objects(labeled_mask)
        
        individual_plants = []

        for i, obj in enumerate(objects):
            if obj is None: continue
            
            # Isola a máscara daquela planta específica
            obj_mask = (labeled_mask[obj] == (i + 1)).astype(np.uint8)
            
            if np.count_nonzero(obj_mask) < min_area:
                continue
                
            individual_plants.append({
                'mask': obj_mask,
                'bbox': obj
            })

        return individual_plants
