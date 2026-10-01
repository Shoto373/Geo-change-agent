"""
Скрипт загрузки и распаковки бенчмарка OSCD (Onera Satellite Change Detection).
Датасет содержит 24 мультиспектральные пары Sentinel-2 (13 каналов) с масками изменений.
Официальный источник: IEEE / ONERA (Onera Satellite Change Detection dataset).
"""

import os
import sys
import zipfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oscd_downloader")

OSCD_URLS = {
    # Ссылки на официальные архивы OSCD
    "images": "http://dase.grss-ieee.org/download/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip",
    "labels": "http://dase.grss-ieee.org/download/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip"
}

def download_file(url: str, dest_path: str):
    logger.info(f"Скачивание: {url} -> {dest_path}")
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\rЗагрузка: {percent:.1f}% ({readsofar / 1024 / 1024:.1f} MB / {totalsize / 1024 / 1024:.1f} MB)"
            sys.stderr.write(s)
            sys.stderr.flush()
        else:
            sys.stderr.write(f"\rЗагружено: {readsofar / 1024 / 1024:.1f} MB")

    urllib.request.urlretrieve(url, dest_path, reporthook)
    sys.stderr.write("\n")

def setup_oscd(data_dir: str = "./data/oscd"):
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Директория для данных: {os.path.abspath(data_dir)}")

    for name, url in OSCD_URLS.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")
        if not os.path.exists(zip_path):
            try:
                download_file(url, zip_path)
            except Exception as e:
                logger.error(f"Не удалось автоматически загрузить {name} ({e}).")
                logger.info(f"Вы можете вручную скачать архив по ссылке: {url} и положить в {zip_path}")
                continue

        logger.info(f"Распаковка {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info(f"Успешно распаковано: {name}")
        except Exception as e:
            logger.error(f"Ошибка распаковки {zip_path}: {e}")

if __name__ == "__main__":
    setup_oscd()
