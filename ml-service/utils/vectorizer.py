"""
Модуль векторизации растровых масок изменений в формат GeoJSON (RFC 7946).
Использует алгоритм Дугласа-Пекера (OpenCV) для аппроксимации контуров и
преобразования пиксельных координат растра в географические координаты WGS84 (Lon/Lat).
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional

def mask_to_geojson_features(
    mask_bool: np.ndarray,
    bbox: List[float],
    min_pixels: int = 15,
    simplify_epsilon: float = 1.5,
    pixel_size_m: float = 10.0,
    properties_extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Преобразует бинарную маску (H, W) и ограничивающий прямоугольник BBox [min_lon, min_lat, max_lon, max_lat]
    в валидный GeoJSON FeatureCollection.

    :param mask_bool: 2D булевый массив (True = обнаружено изменение)
    :param bbox: [min_lon, min_lat, max_lon, max_lat]
    :param min_pixels: минимальное количество пикселей для контура
    :param simplify_epsilon: порог аппроксимации контуров в пикселях
    :param pixel_size_m: размер пикселя в метрах (по умолчанию 10м для Sentinel-2)
    :param properties_extra: дополнительные метаданные для каждого полигона
    :return: GeoJSON словарь (FeatureCollection)
    """
    h, w = mask_bool.shape[:2]
    min_lon, min_lat, max_lon, max_lat = bbox

    # 1. Бинаризация в uint8 для OpenCV
    binary_mask = (mask_bool.astype(np.uint8) * 255)

    # 2. Поиск внешних контуров
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    poly_id = 1

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_pixels:
            continue

        # 3. Аппроксимация полигона алгоритмом Дугласа-Пекера
        approx = cv2.approxPolyDP(cnt, epsilon=simplify_epsilon, closed=True)
        if len(approx) < 3:
            continue

        # 4. Преобразование пиксельных координат в WGS84
        # x -> lon: min_lon + (x / w) * (max_lon - min_lon)
        # y -> lat: max_lat - (y / h) * (max_lat - min_lat) (ось Y инвертирована)
        wgs_coords = []
        for pt in approx:
            px_x, px_y = float(pt[0][0]), float(pt[0][1])
            lon = round(min_lon + (px_x / w) * (max_lon - min_lon), 6)
            lat = round(max_lat - (px_y / h) * (max_lat - min_lat), 6)
            wgs_coords.append([lon, lat])

        # Замыкаем кольцо полигона (первая и последняя точка совпадают)
        if wgs_coords and wgs_coords[0] != wgs_coords[-1]:
            wgs_coords.append(wgs_coords[0])

        if len(wgs_coords) < 4:
            continue

        # 5. Расчет физических метрик полигона
        area_ha = round((area_px * (pixel_size_m * pixel_size_m)) / 10000.0, 3)
        perimeter_m = round(cv2.arcLength(cnt, closed=True) * pixel_size_m, 1)

        # Центроид контура
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            cx_px = moments["m10"] / moments["m00"]
            cy_px = moments["m01"] / moments["m00"]
            cx_lon = round(min_lon + (cx_px / w) * (max_lon - min_lon), 6)
            cy_lat = round(max_lat - (cy_px / h) * (max_lat - min_lat), 6)
            centroid = [cx_lon, cy_lat]
        else:
            centroid = wgs_coords[0]

        props = {
            "polygon_id": poly_id,
            "area_hectares": area_ha,
            "perimeter_meters": perimeter_m,
            "centroid": centroid,
            "pixels_count": int(area_px)
        }
        if properties_extra:
            props.update(properties_extra)

        features.append({
            "type": "Feature",
            "id": poly_id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [wgs_coords]
            },
            "properties": props
        })
        poly_id += 1

    return {
        "type": "FeatureCollection",
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "metadata": {
            "total_features": len(features),
            "total_area_hectares": round(sum(f["properties"]["area_hectares"] for f in features), 2),
            "crs": "EPSG:4326 (WGS84)",
            "generator": "TerraVision AI Vector Engine v2.5"
        },
        "features": features
    }
