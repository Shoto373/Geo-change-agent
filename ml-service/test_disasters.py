import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from detector import analyze_earth_changes

def create_synthetic_forest(size=256):
    np.random.seed(42)
    return {
        "B02": np.random.normal(300, 30, (size, size)),   # Blue
        "B03": np.random.normal(550, 40, (size, size)),   # Green
        "B04": np.random.normal(350, 30, (size, size)),   # Red
        "B08": np.random.normal(3200, 150, (size, size)), # NIR (высокий для здоровой листвы)
        "B12": np.random.normal(800, 60, (size, size))    # SWIR (низкий для влажного леса)
    }

def test_flood_scenario():
    print("\n--- [1/3] ТЕСТИРОВАНИЕ СЦЕНАРИЯ НАВОДНЕНИЯ / ПАВОДКА ---")
    t1 = create_synthetic_forest()
    t2 = create_synthetic_forest()

    # Моделируем разлив реки / затопление поймы (круг в центре)
    y, x = np.ogrid[:256, :256]
    flood_zone = ((x - 128)**2 + (y - 128)**2) <= (50**2)

    # Вода: сильно поглощает NIR и SWIR, отражает в Blue и Green
    t2["B02"][flood_zone] = 1200 # Blue
    t2["B03"][flood_zone] = 1100 # Green (высокий)
    t2["B04"][flood_zone] = 400  # Red
    t2["B08"][flood_zone] = 200  # NIR (почти полный ноль!)
    t2["B12"][flood_zone] = 100  # SWIR

    mask, stats = analyze_earth_changes(t1, t2)
    print(f"Категория: {stats['category']}")
    print(f"Степень:   {stats['severity']}")
    print(f"Площадь:   {stats['total_area_hectares']} га")
    print(f"Отчет:     {stats['description_ru']}")
    assert stats["category"] == "FLOOD", f"Expected FLOOD, got {stats['category']}"
    print("-> Наводнение обнаружено успешно!")

def test_wildfire_scenario():
    print("\n--- [2/3] ТЕСТИРОВАНИЕ СЦЕНАРИЯ ЛЕСНОГО ПОЖАРА / ГАРИ ---")
    t1 = create_synthetic_forest()
    t2 = create_synthetic_forest()

    # Моделируем очаг выгорания (эллипс)
    y, x = np.ogrid[:256, :256]
    fire_zone = ((x - 100)**2 / (60**2) + (y - 120)**2 / (40**2)) <= 1

    # Гарь / Зола: падение NIR, взрывной рост SWIR (B12 отражает сухую золу и минералы)
    t2["B04"][fire_zone] = 900   # Red
    t2["B08"][fire_zone] = 1100  # NIR (упал в 3 раза)
    t2["B12"][fire_zone] = 2800  # SWIR (взлетел до небес!)

    mask, stats = analyze_earth_changes(t1, t2)
    print(f"Категория: {stats['category']}")
    print(f"Степень:   {stats['severity']}")
    print(f"Площадь:   {stats['total_area_hectares']} га")
    print(f"Отчет:     {stats['description_ru']}")
    assert stats["category"] == "WILDFIRE", f"Expected WILDFIRE, got {stats['category']}"
    print("-> Лесной пожар обнаружен успешно!")

def test_deforestation_scenario():
    print("\n--- [3/3] ТЕСТИРОВАНИЕ СЦЕНАРИЯ ВЫРУБКИ ЛЕСА ---")
    t1 = create_synthetic_forest()
    t2 = create_synthetic_forest()

    # Моделируем участок вырубки
    y, x = np.ogrid[:256, :256]
    cut_zone = (x > 50) & (x < 150) & (y > 50) & (y < 120)

    # Обнаженная почва (падение вегетации, без золы и без воды)
    t2["B04"][cut_zone] = 1500  # Red (песок/глина)
    t2["B08"][cut_zone] = 1600  # NIR
    t2["B12"][cut_zone] = 1400  # SWIR

    mask, stats = analyze_earth_changes(t1, t2)
    print(f"Категория: {stats['category']}")
    print(f"Степень:   {stats['severity']}")
    print(f"Площадь:   {stats['total_area_hectares']} га")
    print(f"Отчет:     {stats['description_ru']}")
    assert stats["category"] == "DEFORESTATION", f"Expected DEFORESTATION, got {stats['category']}"
    print("-> Вырубка леса обнаружена успешно!")

if __name__ == "__main__":
    test_flood_scenario()
    test_wildfire_scenario()
    test_deforestation_scenario()
    print("\n ВСЕ ТЕСТЫ ФИЗИКО-СПЕКТРАЛЬНОГО МОНИТОРИНГА УСПЕШНО ПРОЙДЕНЫ!")
