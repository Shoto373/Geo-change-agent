import base64
import io
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector import compute_changes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml_engine")

app = FastAPI(title="Earth Change ML Microservice", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    bbox: List[float]       # [min_lon, min_lat, max_lon, max_lat]
    t1_range: str           # "2021-07-01/2021-08-30"
    t2_range: str           # "2024-07-01/2024-08-30"
    ndvi_threshold: Optional[float] = -0.20
    min_pixels: Optional[int] = 50

def generate_synthetic_scene(bbox: List[float], size: int = 256, is_after: bool = False):
    """
    Генерирует реалистичный синтетический паттерн лесного массива и вырубки
    в случае недоступности внешнего STAC API или тестового режима.
    """
    np.random.seed(42)
    # Фоновый тропический лес: высокий NIR, умеренный Red
    forest_red = np.random.normal(300, 40, (size, size))
    forest_nir = np.random.normal(3200, 200, (size, size))
    forest_green = np.random.normal(550, 50, (size, size))
    forest_blue = np.random.normal(250, 30, (size, size))

    if is_after:
        # Создаем полигон вырубки в центре
        y, x = np.ogrid[:size, :size]
        cut_mask = ((x - 140)**2 / (65**2) + (y - 120)**2 / (45**2)) <= 1
        cut_mask |= ((x - 90)**2 / (30**2) + (y - 160)**2 / (25**2)) <= 1
        
        # Обнаженная почва: низкий NIR, высокий Red
        forest_red[cut_mask] = np.random.normal(1600, 100, np.sum(cut_mask))
        forest_nir[cut_mask] = np.random.normal(1400, 120, np.sum(cut_mask))
        forest_green[cut_mask] = np.random.normal(1100, 80, np.sum(cut_mask))
        forest_blue[cut_mask] = np.random.normal(800, 60, np.sum(cut_mask))

    return {
        "B02": np.clip(forest_blue, 0, 10000),
        "B03": np.clip(forest_green, 0, 10000),
        "B04": np.clip(forest_red, 0, 10000),
        "B08": np.clip(forest_nir, 0, 10000)
    }

def fetch_stac_scene(bbox: List[float], date_range: str, is_after: bool):
    """
    Пытается загрузить реальные данные Sentinel-2 L2A через Planetary Computer,
    при сбое или превышении таймаута переключается на синтетический лес.
    """
    try:
        import pystac_client
        import planetary_computer
        import odc.stac
        from shapely.geometry import box

        logger.info(f"Запрос STAC Planetary Computer для {date_range}...")
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )
        geom = box(*bbox)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=geom,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 20}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            limit=5
        )
        items = list(search.items())
        if items:
            logger.info(f"Найдено {len(items)} сцен. Загрузка лучшей сцены...")
            ds = odc.stac.load(
                [items[0]],
                bands=["B02", "B03", "B04", "B08"],
                bbox=bbox,
                resolution=10,
                chunks={"x": 512, "y": 512}
            ).isel(time=0)
            
            return {
                "B02": ds["B02"].values,
                "B03": ds["B03"].values,
                "B04": ds["B04"].values,
                "B08": ds["B08"].values,
                "source": f"Sentinel-2 L2A ({items[0].id})"
            }
        else:
            logger.warning(f"Сцены с облачностью < 20% не найдены для {date_range}. Активирован Synthetic Fallback.")
    except Exception as e:
        logger.warning(f"STAC fetch warning: {e}. Переключение на Synthetic Fallback.")

    data = generate_synthetic_scene(bbox, size=256, is_after=is_after)
    data["source"] = "Synthetic High-Fidelity Amazon Forest"
    return data

def raster_to_png_b64(r: np.ndarray, g: np.ndarray, b: np.ndarray, overlay_mask: Optional[np.ndarray] = None) -> str:
    """Конвертирует каналы в нормализованный Base64 PNG."""
    rgb = np.stack([r, g, b], axis=-1).astype(float)
    rgb = np.clip(rgb / 3000.0, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(rgb)
    
    if overlay_mask is not None and np.any(overlay_mask):
        masked = np.ma.masked_where(~overlay_mask, overlay_mask)
        ax.imshow(masked, cmap="autumn", alpha=0.65, interpolation="none")

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/health")
def health():
    return {"status": "ok", "service": "earth-change-ml-engine"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if len(req.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must have 4 elements: [min_lon, min_lat, max_lon, max_lat]")

    logger.info(f"Запуск анализа для BBox: {req.bbox}")
    t1_data = fetch_stac_scene(req.bbox, req.t1_range, is_after=False)
    t2_data = fetch_stac_scene(req.bbox, req.t2_range, is_after=True)

    mask, conf_map, stats = compute_changes(
        t1_red=t1_data["B04"],
        t1_nir=t1_data["B08"],
        t2_red=t2_data["B04"],
        t2_nir=t2_data["B08"],
        ndvi_threshold=req.ndvi_threshold,
        min_pixels=req.min_pixels
    )

    t1_rgb_b64 = raster_to_png_b64(t1_data["B04"], t1_data["B03"], t1_data["B02"])
    t2_rgb_b64 = raster_to_png_b64(t2_data["B04"], t2_data["B03"], t2_data["B02"])
    mask_overlay_b64 = raster_to_png_b64(t2_data["B04"], t2_data["B03"], t2_data["B02"], overlay_mask=mask)

    return {
        "status": "success",
        "bbox": req.bbox,
        "data_source": {
            "t1": t1_data.get("source"),
            "t2": t2_data.get("source")
        },
        "stats": stats,
        "images": {
            "t1_rgb": f"data:image/png;base64,{t1_rgb_b64}",
            "t2_rgb": f"data:image/png;base64,{t2_rgb_b64}",
            "mask_overlay": f"data:image/png;base64,{mask_overlay_b64}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
