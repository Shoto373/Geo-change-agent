import numpy as np
from scipy import ndimage
from typing import Dict, Any, Tuple, Optional

def compute_spectral_indices(
    green: np.ndarray,
    red: np.ndarray, 
    nir: np.ndarray, 
    swir: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Вычисляет физические спектральные индексы Sentinel-2:
      1. NDVI (Normalized Difference Vegetation Index) - вегетация и биомасса.
      2. NDWI (Normalized Difference Water Index) - водные объекты и затопления.
      3. NBR  (Normalized Burn Ratio) - выгорание, лесные пожары и гари.
    """
    # 1. NDVI = (NIR - Red) / (NIR + Red)
    denom_ndvi = nir + red
    denom_ndvi = np.where(denom_ndvi == 0, 1e-6, denom_ndvi)
    ndvi = (nir - red) / denom_ndvi

    # 2. NDWI = (Green - NIR) / (Green + NIR)
    # Вода имеет положительный NDWI, суша и растительность - отрицательный.
    denom_ndwi = green + nir
    denom_ndwi = np.where(denom_ndwi == 0, 1e-6, denom_ndwi)
    ndwi = (green - nir) / denom_ndwi

    # 3. NBR = (NIR - SWIR) / (NIR + SWIR)
    nbr = None
    if swir is not None:
        denom_nbr = nir + swir
        denom_nbr = np.where(denom_nbr == 0, 1e-6, denom_nbr)
        nbr = (nir - swir) / denom_nbr

    return {
        "ndvi": np.clip(ndvi, -1.0, 1.0),
        "ndwi": np.clip(ndwi, -1.0, 1.0),
        "nbr": np.clip(nbr, -1.0, 1.0) if nbr is not None else None
    }

def analyze_earth_changes(
    t1_bands: Dict[str, np.ndarray],
    t2_bands: Dict[str, np.ndarray],
    pixel_size_m: float = 10.0,
    min_cluster_pixels: int = 25
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Комплексный физико-спектральный классификатор изменений (Sentinel-2 L2A).
    Автоматически распознает:
      1. Наводнения и паводки (ΔNDWI > +0.22, появление открытой воды).
      2. Лесные пожары и гари (ΔNBR > +0.25, термическое поражение SWIR).
      3. Вырубки и обезлесение (ΔNDVI < -0.20, утрата кроны).
      4. Инфраструктурные изменения и строительство.
    """
    h, w = t1_bands["B04"].shape
    pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0

    # Вычисляем индексы для обеих дат
    idx1 = compute_spectral_indices(
        green=t1_bands["B03"],
        red=t1_bands["B04"],
        nir=t1_bands["B08"],
        swir=t1_bands.get("B12")
    )
    idx2 = compute_spectral_indices(
        green=t2_bands["B03"],
        red=t2_bands["B04"],
        nir=t2_bands["B08"],
        swir=t2_bands.get("B12")
    )

    d_ndvi = idx2["ndvi"] - idx1["ndvi"]
    d_ndwi = idx2["ndwi"] - idx1["ndwi"]
    d_nbr = (idx1["nbr"] - idx2["nbr"]) if (idx1["nbr"] is not None and idx2["nbr"] is not None) else None

    # Пороговые маски явлений
    # 1. Затопление (вода появилась там, где её не было)
    flood_candidates = (d_ndwi > 0.20) & (idx2["ndwi"] > -0.05)

    # 2. Пожар / Гарь: падение d_nbr > 0.27 и отрицательный NBR на T2 (отклик золы: SWIR > NIR)
    if d_nbr is not None:
        fire_candidates = (d_nbr > 0.25) & (idx2["nbr"] < 0.05)
    else:
        # Fallback при отсутствии SWIR: резкое потемнение и обугливание (падение NIR ниже Red)
        fire_candidates = (d_ndvi < -0.35) & (t2_bands["B04"] > t2_bands["B08"])

    # 3. Вырубка леса (потеря вегетации без затопления и без горения)
    deforest_candidates = (d_ndvi < -0.22) & ~flood_candidates & ~fire_candidates

    # Объединенная маска любых изменений
    combined_raw = flood_candidates | fire_candidates | deforest_candidates

    # Фильтрация шумов через связанность компонент (connected components)
    labeled, num_features = ndimage.label(combined_raw)
    if num_features > 0:
        sizes = ndimage.sum(combined_raw, labeled, range(num_features + 1))
        clean_mask = combined_raw.copy()
        clean_mask[sizes[labeled] < min_cluster_pixels] = False
    else:
        clean_mask = np.zeros((h, w), dtype=bool)

    total_changed_pixels = int(np.sum(clean_mask))
    total_area_ha = round(total_changed_pixels * pixel_area_ha, 2)

    # Определение доминирующего типа происшествия
    n_flood = int(np.sum(flood_candidates & clean_mask))
    n_fire = int(np.sum(fire_candidates & clean_mask))
    n_deforest = int(np.sum(deforest_candidates & clean_mask))

    if total_changed_pixels == 0 or total_area_ha < 0.2:
        category = "NO_CHANGE"
        severity = "НЕТ"
        description_ru = "На снимках изменений не обнаружено. Обстановка на местности стабильна."
    elif n_flood > max(n_fire, n_deforest):
        category = "FLOOD"
        severity = "ВЫСОКАЯ" if total_area_ha > 50 else ("СРЕДНЯЯ" if total_area_ha > 10 else "ЛОКАЛЬНАЯ")
        description_ru = (
            f"Зафиксирована зона паводкового затопления (наводнения). "
            f"Площадь зеркала воды увеличилась на {total_area_ha} га. Степень: {severity.lower()}."
        )
    elif n_fire > max(n_flood, n_deforest):
        category = "WILDFIRE"
        severity = "КРИТИЧЕСКАЯ (ВЫСОКОЕ ВЫГОРАНИЕ)" if total_area_ha > 100 else ("СРЕДНЯЯ ГАРЬ" if total_area_ha > 20 else "НИЗОВАЯ ГАРЬ")
        description_ru = (
            f"Обнаружен очаг лесного пожара / свежая гарь (спектральный отклик золы). "
            f"Площадь термопоражения: {total_area_ha} га. Категория: {severity.lower()}."
        )
    elif n_deforest > 0:
        category = "DEFORESTATION"
        severity = "МАССИВНАЯ РАСЧИСТКА" if total_area_ha > 25 else "ВЫРУБКА УЧАСТКА"
        description_ru = (
            f"Зафиксирована потеря лесного полога (вырубка / расчистка растительности). "
            f"Общая площадь затронутого лесного массива: {total_area_ha} га."
        )
    else:
        category = "LAND_COVER_CHANGE"
        severity = "УМЕРЕННАЯ"
        description_ru = f"Зафиксированы изменения почвенно-растительного покрова на площади {total_area_ha} га."

    stats = {
        "category": category,
        "severity": severity,
        "description_ru": description_ru,
        "changed_pixels": total_changed_pixels,
        "total_area_hectares": total_area_ha,
        "breakdown": {
            "flood_hectares": round(n_flood * pixel_area_ha, 2),
            "wildfire_hectares": round(n_fire * pixel_area_ha, 2),
            "deforestation_hectares": round(n_deforest * pixel_area_ha, 2)
        },
        "indices_delta": {
            "mean_delta_ndvi": round(float(np.mean(d_ndvi)), 3),
            "mean_delta_ndwi": round(float(np.mean(d_ndwi)), 3),
            "mean_delta_nbr": round(float(np.mean(d_nbr)), 3) if d_nbr is not None else None
        }
    }

    return clean_mask, stats
