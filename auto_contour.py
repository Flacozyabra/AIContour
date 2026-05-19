#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Скрипт автоматического оконтуривания органов риска (OAR) на КТ-исследованиях
================================================================================
Этот скрипт является MVP для сегментирования анатомических структур на КТ-снимках
и их последующего экспорта в формат DICOM RTSTRUCT для систем планирования (TPS).

Особенности:
1. Оптимизирован для работы на ПК с 16 ГБ ОЗУ (использует CPU и быстрый режим 3мм).
2. Защищен от утечек памяти с помощью принудительного вызова сборщика мусора.
3. Имеет гибкую систему пресетов для выбора OAR.

--------------------------------------------------------------------------------
Инструкция по развертыванию и установке зависимостей (Windows PowerShell):
--------------------------------------------------------------------------------
1. Создайте виртуальное окружение:
   python -m venv venv

2. Активируйте виртуальное окружение:
   .\venv\Scripts\Activate.ps1

3. Обновите pip до последней версии:
   python -m pip install --upgrade pip

4. Установите необходимые библиотеки:
   pip install numpy pydicom dicom2nifti totalsegmentator rt-utils nibabel

5. Запуск скрипта:
   python auto_contour.py --input "C:\path\to\ct_dicom" --output "C:\path\to\output_dir" --preset abdominal_oar
================================================================================
"""

import os
import sys
import gc
import time
import argparse
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Настройка логирования на русском языке
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AutoContour")

# Пресеты органов риска (OAR) для оптимизации экспорта
PRESETS: Dict[str, List[str]] = {
    "abdominal_oar": [
        "spleen",            # Селезенка
        "kidney_right",      # Правая почка
        "kidney_left",       # Левая почка
        "gallbladder",       # Желчный пузырь
        "liver",             # Печень
        "stomach",           # Желудок
        "aorta",             # Аорта
        "inferior_vena_cava",# Нижняя полая вена
        "bladder",           # Мочевой пузырь
        "heart"              # Сердце
    ],
    "thoracic_oar": [
        "heart",             # Сердце
        "lung_left",         # Левое легкое
        "lung_right",        # Правое легкое
        "trachea",           # Трахея
        "aorta",             # Аорта
        "esophagus"          # Пищевод
    ]
}

# Гармоничные цвета для отображения контуров в TPS (формат RGB)
ORGAN_COLORS: Dict[str, List[int]] = {
    "spleen": [156, 39, 176],         # Фиолетовый
    "kidney_right": [3, 169, 244],     # Голубой
    "kidney_left": [33, 150, 243],     # Синий
    "gallbladder": [76, 175, 80],      # Зеленый
    "liver": [139, 195, 74],          # Салатовый
    "stomach": [255, 152, 0],         # Оранжевый
    "aorta": [244, 67, 54],           # Красный
    "inferior_vena_cava": [63, 81, 181], # Темно-синий
    "bladder": [255, 235, 59],        # Желтый
    "heart": [233, 30, 99],           # Розовый
    "lung_left": [0, 150, 136],        # Бирюзовый
    "lung_right": [0, 188, 212],       # Светло-бирюзовый
    "trachea": [121, 85, 72],         # Коричневый
    "esophagus": [158, 158, 158],     # Серый
    "pancreas": [255, 193, 7]          # Янтарный
}


def verify_dicom_directory(dicom_dir: Path) -> int:
    """
    Проверяет корректность входной папки DICOM и считает количество файлов .dcm.

    :param dicom_dir: Путь к папке с DICOM файлами.
    :return: Количество найденных DICOM файлов.
    :raises FileNotFoundError: Если папка пуста или не содержит файлов .dcm.
    """
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"Указанный путь к DICOM не существует или не является папкой: {dicom_dir}")

    dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
    # Если расширения нет, проверим все файлы
    if not dicom_files:
        dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]

    num_files = len(dicom_files)
    if num_files == 0:
        raise FileNotFoundError(f"В папке {dicom_dir} не найдено DICOM-файлов.")

    logger.info(f"Найдено DICOM файлов для обработки: {num_files}")
    return num_files


def run_pipeline(dicom_dir_path: str, output_dir_path: str, preset_name: str) -> None:
    """
    Основной пайплайн выполнения автооконтурирования органов риска на КТ.

    :param dicom_dir_path: Путь к директории с исходными DICOM КТ-слайсами.
    :param output_dir_path: Путь для сохранения итогового rtstruct.dcm.
    :param preset_name: Название используемого пресета OAR.
    """
    start_time = time.time()
    dicom_dir = Path(dicom_dir_path).resolve()
    output_dir = Path(output_dir_path).resolve()
    
    # 0. Инициализация временных путей
    temp_dir = output_dir / "temp_autocontour_workspace"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_ct_path = temp_dir / "temp_ct_volume.nii.gz"
    segmentation_dir = temp_dir / "temp_masks"
    
    try:
        # Проверка DICOM-файлов
        verify_dicom_directory(dicom_dir)
        
        # ----------------------------------------------------------------------
        # Шаг 1: Конвертация DICOM -> NIfTI
        # ----------------------------------------------------------------------
        logger.info("--- Шаг 1 из 5: Конвертация DICOM в 3D NIfTI объем ---")
        import dicom2nifti
        import dicom2nifti.settings as settings
        
        # Отключаем строгие проверки геометрии, чтобы скрипт не падал на клинических КТ
        # с неравномерным шагом, малым числом срезов или небольшим наклоном гентри.
        settings.disable_validate_slice_increment()
        settings.disable_validate_orthogonal()
        settings.disable_validate_orientation()
        
        step_start = time.time()
        logger.info(f"Сборка 3D-тома NIfTI из {dicom_dir}... Это может занять некоторое время.")
        
        # Запуск конвертации без реориентации (reorient_nifti=False), чтобы сохранить
        # оригинальную геометрию DICOM-координат для точного совмещения в rt-utils.
        dicom2nifti.dicom_series_to_nifti(str(dicom_dir), str(nifti_ct_path), reorient_nifti=False)
        
        if not nifti_ct_path.exists():
            raise RuntimeError("Не удалось создать временный NIfTI-файл КТ.")
            
        logger.info(f"Шаг 1 успешно завершен за {time.time() - step_start:.2f} сек.")
        logger.info(f"Временный NIfTI сохранен: {nifti_ct_path} ({nifti_ct_path.stat().st_size / (1024*1024):.2f} МБ)")

        # ----------------------------------------------------------------------
        # Шаг 2: ИИ-сегментация (TotalSegmentator на CPU с оптимизацией)
        # ----------------------------------------------------------------------
        logger.info("--- Шаг 2 из 5: ИИ-сегментация с помощью TotalSegmentator ---")
        from totalsegmentator.python_api import totalsegmentator
        
        step_start = time.time()
        logger.warning(
            "ВНИМАНИЕ: Запуск сегментации принудительно на CPU в быстром режиме (fast=True). "
            "При первом запуске скрипт автоматически скачает веса модели (~150-200 МБ) из сети. "
            "Процесс сегментации на CPU может занять от 2 до 5 минут в зависимости от ПК."
        )
        
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        
        # Запуск TotalSegmentator через официальный Python API
        totalsegmentator(
            input=str(nifti_ct_path),
            output=str(segmentation_dir),
            device="cpu",
            fast=True
        )
        
        logger.info(f"Шаг 2 успешно завершен за {time.time() - step_start:.2f} сек.")

        # ----------------------------------------------------------------------
        # Шаг 3: Принудительная очистка памяти (ОЗУ)
        # ----------------------------------------------------------------------
        logger.info("--- Шаг 3 из 5: Выгрузка моделей и принудительная очистка ОЗУ ---")
        step_start = time.time()
        
        # Удаляем тяжелые ссылки, чтобы вызвать сборщик мусора
        if 'totalsegmentator' in sys.modules:
            # Некоторые кэши PyTorch или библиотеки ИИ могут удерживать память на CPU
            import torch
            torch.cuda.empty_cache()  # На случай если GPU частично упоминался
            
        # Удаляем временный файл КТ для экономии диска перед сборкой
        if nifti_ct_path.exists():
            nifti_ct_path.unlink()
            
        gc.collect()
        logger.info(f"Шаг 3 успешно завершен за {time.time() - step_start:.2f} сек. Память очищена.")

        # ----------------------------------------------------------------------
        # Шаг 4: Сборка масок в DICOM RTSTRUCT
        # ----------------------------------------------------------------------
        logger.info("--- Шаг 4 из 5: Сборка RTSTRUCT и привязка к геометрии DICOM ---")
        import nibabel as nib
        import numpy as np
        from rt_utils import RTStructBuilder
        
        step_start = time.time()
        
        # Получаем список масок из папки сегментации
        mask_files = list(segmentation_dir.glob("*.nii.gz"))
        if not mask_files:
            raise RuntimeError("Не найдено масок органов после сегментации.")
            
        logger.info(f"Обнаружено сегментированных масок органов: {len(mask_files)}")
        
        # Загружаем выбранный пресет органов
        target_organs = PRESETS.get(preset_name)
        if target_organs:
            logger.info(f"Используется пресет '{preset_name}'. Фильтрация OAR: {target_organs}")
        else:
            logger.warning(f"Пресет '{preset_name}' не найден. Будут экспортированы все найденные OAR.")
            target_organs = None
            
        # Инициализируем новый RTSTRUCT на основе оригинальных КТ-слайсов
        logger.info("Инициализация RTSTRUCT считыванием оригинальной геометрии DICOM серии...")
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_dir))
        
        added_count = 0
        for mask_file in mask_files:
            organ_name = mask_file.name.replace(".nii.gz", "")
            
            # Фильтруем органы по пресету, если он задан
            if target_organs and organ_name not in target_organs:
                continue
                
            logger.info(f"Обработка органа: {organ_name}...")
            
            # Загружаем NIfTI маску
            nii_mask = nib.load(str(mask_file))
            mask_data = nii_mask.get_fdata() > 0.5  # Приведение к бинарной bool маске
            
            # В NIfTI оси ориентированы как (X, Y, Z) [Cols, Rows, Slices].
            # Библиотека rt-utils ожидает (Rows, Cols, Slices), что соответствует (Y, X, Z) в NumPy.
            # Для этого транспонируем первые две оси.
            mask_data_transposed = np.transpose(mask_data, (1, 0, 2))
            
            # Преобразуем к типу bool для rt-utils
            mask_bool = mask_data_transposed.astype(bool)
            
            # Если маска пустая (орган не попал в КТ), пропускаем его для чистоты RTSTRUCT
            if not np.any(mask_bool):
                logger.info(f"Пропуск пустого органа: {organ_name} (отсутствует в КТ объеме)")
                continue
                
            # Выбираем цвет для отображения
            color = ORGAN_COLORS.get(organ_name, [128, 128, 128])  # Серый по умолчанию
            
            # Создаем красивое имя
            pretty_name = organ_name.replace("_", " ").title()
            
            # Добавляем ROI в структуру
            rtstruct.add_roi(
                mask=mask_bool,
                color=color,
                name=pretty_name
            )
            added_count += 1
            logger.info(f"Успешно добавлен ROI '{pretty_name}' (цвет: {color})")
            
        if added_count == 0:
            raise RuntimeError("В RTSTRUCT не было добавлено ни одного OAR. Проверьте соответствие КТ области выбранному пресету.")
            
        # ----------------------------------------------------------------------
        # Шаг 5: Сохранение итогового файла
        # ----------------------------------------------------------------------
        logger.info("--- Шаг 5 из 5: Запись итогового DICOM RTSTRUCT ---")
        output_dir.mkdir(parents=True, exist_ok=True)
        rtstruct_file_path = output_dir / "rtstruct.dcm"
        
        rtstruct.save(str(rtstruct_file_path))
        logger.info(f"Шаг 5 успешно завершен за {time.time() - step_start:.2f} сек.")
        logger.info(f"Итоговый файл RTSTRUCT успешно записан: {rtstruct_file_path}")
        
    except Exception as e:
        logger.error(f"Произошел критический сбой во время выполнения пайплайна: {e}", exc_info=True)
        raise e
        
    finally:
        # Полная очистка временных папок после выполнения (или сбоя)
        logger.info("Очистка временных папок и файлов...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.info(f"Пайплайн полностью завершен. Общее время работы: {time.time() - start_time:.2f} сек.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MVP автооконтурирования органов риска на КТ для лучевой терапии."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Путь к папке, содержащей исходные КТ-срезы в формате DICOM."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Путь к папке, в которую будет сохранен готовый файл rtstruct.dcm."
    )
    parser.add_argument(
        "-p", "--preset",
        default="abdominal_oar",
        choices=list(PRESETS.keys()) + ["all"],
        help="Пресет органов риска для экспорта (по умолчанию: abdominal_oar)."
    )
    
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.preset)
