"""
Скрипт загрузки и распаковки официального датасета LEVIR-MCI с Hugging Face.
Размер архива: ~2.64 ГБ.
Содержит 10 077 пар снимков A/B, маски label/ и JSON аннотации train/val/test.
"""

import os
import sys
import zipfile
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("levir_mci_downloader")

def download_and_extract_levir_mci(target_dir: str = "./data/levir_mci"):
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Целевая папка для датасета: {os.path.abspath(target_dir)}")

    logger.info("Запуск загрузки LEVIR-MCI-dataset.zip (2.64 ГБ) из Hugging Face (lcybuaa/LEVIR-MCI)...")
    try:
        zip_path = hf_hub_download(
            repo_id="lcybuaa/LEVIR-MCI",
            filename="LEVIR-MCI-dataset.zip",
            repo_type="dataset",
            local_dir=target_dir
        )
        logger.info(f"Архив успешно загружен: {zip_path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки через huggingface_hub: {e}")
        return False

    logger.info("Распаковка архива LEVIR-MCI-dataset.zip (это займет 1-2 минуты)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        logger.info("Распаковка успешно завершена!")
        
        # Удаляем архив после успешной распаковки для экономии места
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("Архив .zip удален, папки с данными готовы.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при распаковке архива: {e}")
        return False

if __name__ == "__main__":
    download_and_extract_levir_mci()
