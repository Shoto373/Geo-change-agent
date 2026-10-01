import numpy as np
from scipy import ndimage
from typing import Dict, Any, Tuple

def compute_spectral_indices(
    red: np.ndarray, 
    nir: np.ndarray, 
    swir: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет NDVI (Normalized Difference Vegetation Index)
    и NBR (Normalized Burn Ratio / Moisture Index).
    """
    # NDVI = (NIR - Red) / (NIR + Red)
    denom_ndvi = nir + red
    denom_ndvi[denom_ndvi == 0] = 1e-6
    ndvi = (nir - red) / denom_ndvi

    nbr = None
    if swir is not None:
        # NBR = (NIR - SWIR) / (NIR + SWIR)
        denom_nbr = nir + swir
        denom_nbr[denom_nbr == 0] = 1e-6
        nbr = (nir - swir) / denom_nbr

    return ndvi, nbr

def compute_changes(
    t1_red: np.ndarray,
    t1_nir: np.ndarray,
    t2_red: np.ndarray,
    t2_nir: np.ndarray,
    ndvi_threshold: float = -0.20,
    min_pixels: int = 50,
    pixel_size_m: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Анализирует динамику вегетации между T1 и T2.

    :param t1_red: Красный канал на дату T1 (H, W)
    :param t1_nir: Ближний ИК канал на дату T1 (H, W)
    :param t2_red: Красный канал на дату T2 (H, W)
    :param t2_nir: Ближний ИК канал на дату T2 (H, W)
    :param ndvi_threshold: Порог падения NDVI для классификации потери покрова
    :param min_pixels: Минимальное количество пикселей в кластере (50 пикс по 10м = 0.5 га)
    :param pixel_size_m: Размер пикселя в метрах (10м для Sentinel-2)
    :return: (бинарная_маска, карта_уверенности, словарь_статистики)
    """
    ndvi1, _ = compute_spectral_indices(t1_red, t1_nir)
    ndvi2, _ = compute_spectral_indices(t2_red, t2_nir)

    delta_ndvi = ndvi2 - ndvi1

    # Сырая маска потери растительности
    raw_mask = delta_ndvi < ndvi_threshold

    # Морфологическая фильтрация шума
    labeled, num_features = ndimage.label(raw_mask)
    if num_features > 0:
        sizes = ndimage.sum(raw_mask, labeled, range(num_features + 1))
        clean_mask = raw_mask.copy()
        clean_mask[sizes[labeled] < min_pixels] = False
    else:
        clean_mask = np.zeros_like(raw_mask, dtype=bool)

    changed_pixels = int(np.sum(clean_mask))
    pixel_area_m2 = pixel_size_m * pixel_size_m
    total_area_ha = float(changed_pixels * pixel_area_m2 / 10000.0)

    # Карта уверенности: нормализованное отклонение от порога
    confidence_map = np.clip(np.abs(delta_ndvi - ndvi_threshold) / 0.4, 0.0, 1.0)
    confidence_map = np.where(clean_mask, confidence_map, 0.0)

    mean_conf = float(np.mean(confidence_map[clean_mask])) if changed_pixels > 0 else 0.0

    stats = {
        "changed_pixels": changed_pixels,
        "changed_area_hectares": round(total_area_ha, 2),
        "mean_delta_ndvi": float(np.mean(delta_ndvi)),
        "mean_confidence": round(mean_conf, 3),
        "is_deforestation_detected": bool(total_area_ha >= 0.5)
    }

    return clean_mask, confidence_map, stats
