#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Скрипт автоматического оконтурирования органов риска (OAR) на КТ-исследованиях
================================================================================
Этот скрипт является MVP для сегментирования анатомических структур на КТ-снимках
и их последующего экспорта в формат DICOM RTSTRUCT для систем планирования (TPS).

Особенности:
1. Поддерживает два режима: графический интерфейс (GUI) и командную строку (CLI).
2. Защищен от утечек памяти с помощью принудительного вызова сборщика мусора.
3. Имеет гибкую систему пресетов для выбора OAR.
4. Выполняет нейросетевую обработку в отдельном потоке (GUI не зависает).
5. Сканирует КТ на наличие существующей разметки врача и позволяет слить контуры.

--------------------------------------------------------------------------------
Инструкция по установке зависимостей (Windows PowerShell):
--------------------------------------------------------------------------------
    pip install PyQt6 pydicom dicom2nifti totalsegmentator rt-utils nibabel
================================================================================
"""

import os
# Предотвращение крашей из-за дублирования библиотек OpenMP на Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
import time
import math
import argparse
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

# Предварительный импорт необходимых библиотек на главном потоке
import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti
import dicom2nifti.settings as settings
from rt_utils import RTStructBuilder


# Импорт PyQt6 для GUI режима
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QListWidget, QListWidgetItem,
        QRadioButton, QButtonGroup, QTextEdit, QProgressBar, QFileDialog,
        QMessageBox, QFrame, QSplitter, QCheckBox, QDialog, QTextBrowser
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QSettings, QTimer
    from PyQt6.QtGui import QTextCursor, QBrush, QColor, QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Настройка логирования на русском языке
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AutoContour")
PRESETS: Dict[str, List[str]] = {
    "head_neck_oar": [
        "brain",                           # Головной мозг
        "spinal_cord",                     # Спинной мозг
        "thyroid_gland",                   # Щитовидная железа
        "skull",                           # Череп
        "trachea",                         # Трахея
        "esophagus",                       # Пищевод
        "common_carotid_artery_left",      # Левая сонная артерия
        "common_carotid_artery_right"      # Правая сонная артерия
    ],
    "abdominal_oar": [
        "spleen",                          # Селезенка
        "kidney_right",                    # Правая почка
        "kidney_left",                     # Левая почка
        "gallbladder",                     # Желчный пузырь
        "liver",                           # Печень
        "stomach",                         # Желудок
        "aorta",                           # Аорта
        "inferior_vena_cava",              # Нижняя полая вена
        "urinary_bladder",                 # Мочевой пузырь
        "heart",                           # Сердце
        "pancreas",                        # Поджелудочная железа
        "duodenum",                        # Двенадцатиперстная кишка
        "adrenal_gland_left",              # Левый надпочечник
        "adrenal_gland_right",             # Правый надпочечник
        "portal_vein_and_splenic_vein"     # Воротная/селезеночная вена
    ],
    "thoracic_oar": [
        "heart",                           # Сердце
        "lung_left",                       # Левое легкое
        "lung_right",                      # Правое легкое
        "trachea",                         # Трахея
        "aorta",                           # Аорта
        "esophagus",                       # Пищевод
        "pulmonary_artery",                # Легочная артерия
        "superior_vena_cava",              # Верхняя полая вена
        "sternum",                         # Грудина
        "clavicula_left",                  # Левая ключица
        "clavicula_right"                  # Правая ключица
    ],
    "pelvis_oar": [
        "urinary_bladder",                 # Мочевой пузырь
        "prostate",                        # Предстательная железа
        "rectum",                          # Прямая кишка
        "colon",                           # Кишечник
        "small_bowel",                     # Тонкая кишка
        "femur_left",                      # Левая бедренная кость
        "femur_right",                     # Правая бедренная кость
        "hip_left",                        # Левая тазовая кость
        "hip_right",                       # Правая тазовая кость
        "sacrum",                          # Крестец
        "iliac_artery_left",               # Левая подвздошная артерия
        "iliac_artery_right"               # Правая подвздошная артерия
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
    "urinary_bladder": [255, 235, 59],  # Желтый
    "heart": [233, 30, 99],           # Розовый
    "lung_left": [0, 150, 136],        # Бирюзовый
    "lung_right": [0, 188, 212],       # Светло-бирюзовый
    "trachea": [121, 85, 72],         # Коричневый
    "esophagus": [158, 158, 158],     # Серый
    "pancreas": [255, 193, 7],         # Янтарный
    "duodenum": [173, 20, 87],         # Темно-розовый
    "adrenal_gland_left": [255, 87, 34], # Ярко-оранжевый
    "adrenal_gland_right": [255, 112, 67], # Светло-оранжевый
    "pulmonary_artery": [0, 150, 255], # Ярко-голубой
    "small_bowel": [103, 58, 183],     # Темно-фиолетовый
    "prostate": [233, 30, 99],         # Розовый
    "rectum": [121, 85, 72],           # Коричневый
    "colon": [0, 121, 107],            # Темно-бирюзовый
    "femur_left": [255, 224, 178],     # Светло-оранжевый
    "femur_right": [255, 224, 178],    # Светло-оранжевый
    "hip_left": [230, 238, 156],       # Салатово-желтый
    "hip_right": [230, 238, 156],      # Салатово-желтый
    "sacrum": [141, 110, 99],          # Серо-коричневый
    "spinal_cord": [0, 255, 0],        # Зеленый
    "thyroid_gland": [255, 105, 180],  # Розовый
    "skull": [255, 228, 196],          # Бежевый
    "brain": [135, 206, 250],          # Небесно-голубой
    "common_carotid_artery_left": [220, 20, 60],      # Малиновый
    "common_carotid_artery_right": [220, 20, 60],     # Малиновый
    "superior_vena_cava": [70, 130, 180],              # Стальной синий
    "portal_vein_and_splenic_vein": [0, 139, 139],     # Темно-бирюзовый
    "clavicula_left": [244, 164, 96],                  # Песочно-коричневый
    "clavicula_right": [244, 164, 96],                 # Песочно-коричневый
    "sternum": [222, 184, 135],                        # Древесный
    "iliac_artery_left": [255, 99, 71],                # Томатный
    "iliac_artery_right": [255, 99, 71]                # Томатный
}

# Полный перечень всех OAR, доступных в интерфейсе
ALL_ORGANS = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
    "stomach", "aorta", "inferior_vena_cava", "urinary_bladder", "heart",
    "lung_left", "lung_right", "trachea", "esophagus", "pancreas",
    "duodenum", "adrenal_gland_left", "adrenal_gland_right", "pulmonary_artery",
    "small_bowel", "prostate", "rectum", "colon", "femur_left", "femur_right",
    "hip_left", "hip_right", "sacrum", "spinal_cord", "thyroid_gland", "skull",
    "brain", "common_carotid_artery_left", "common_carotid_artery_right",
    "superior_vena_cava", "portal_vein_and_splenic_vein",
    "clavicula_left", "clavicula_right", "sternum",
    "iliac_artery_left", "iliac_artery_right"
]

# Отображаемые на русском языке имена для списка интерфейса
ORGAN_RU_NAMES = {
    "spleen": "Селезенка (Spleen)",
    "kidney_right": "Правая почка (Kidney R)",
    "kidney_left": "Левая почка (Kidney L)",
    "gallbladder": "Желчный пузырь (Gallbladder)",
    "liver": "Печень (Liver)",
    "stomach": "Желудок (Stomach)",
    "aorta": "Аорта (Aorta)",
    "inferior_vena_cava": "Нижняя полая вена (Vena Cava)",
    "urinary_bladder": "Мочевой пузырь (Bladder)",
    "heart": "Сердце (Heart)",
    "lung_left": "Левое легкое (Lung L)",
    "lung_right": "Правое легкое (Lung R)",
    "trachea": "Трахея (Trachea)",
    "esophagus": "Пищевод (Esophagus)",
    "pancreas": "Поджелудочная железа (Pancreas)",
    "duodenum": "Двенадцатиперстная кишка (Duodenum)",
    "adrenal_gland_left": "Левый надпочечник (Adrenal Gland L)",
    "adrenal_gland_right": "Правый надпочечник (Adrenal Gland R)",
    "pulmonary_artery": "Легочная артерия (Pulmonary Artery)",
    "small_bowel": "Тонкая кишка (Small Bowel)",
    "prostate": "Предстательная железа (Prostate)",
    "rectum": "Прямая кишка (Rectum)",
    "colon": "Кишечник (Colon)",
    "femur_left": "Левое бедро (Femur L)",
    "femur_right": "Правое бедро (Femur R)",
    "hip_left": "Левый таз (Hip L)",
    "hip_right": "Правый таз (Hip R)",
    "sacrum": "Крестец (Sacrum)",
    "spinal_cord": "Спинной мозг (Spinal Cord)",
    "thyroid_gland": "Щитовидная железа (Thyroid Gland)",
    "skull": "Череп (Skull)",
    "brain": "Головной мозг (Brain)",
    "common_carotid_artery_left": "Левая сонная артерия (Carotid A L)",
    "common_carotid_artery_right": "Правая сонная артерия (Carotid A R)",
    "superior_vena_cava": "Верхняя полая вена (Vena Cava Sup)",
    "portal_vein_and_splenic_vein": "Воротная вена (Portal/Splenic V)",
    "clavicula_left": "Левая ключица (Clavicle L)",
    "clavicula_right": "Правая ключица (Clavicle R)",
    "sternum": "Грудина (Sternum)",
    "iliac_artery_left": "Левая подвздошная артерия (Iliac A L)",
    "iliac_artery_right": "Правая подвздошная артерия (Iliac A R)"
}

# Карта пресетов для GUI
PRESETS_MAP = {
    "Голова и шея (Head & Neck)": PRESETS["head_neck_oar"],
    "Грудная клетка (Thorax)": PRESETS["thoracic_oar"],
    "Брюшная полость (Abdomen)": PRESETS["abdominal_oar"],
    "Малый таз (Pelvis)": PRESETS["pelvis_oar"],
    "Все органы (All)": ALL_ORGANS,
    "Пользовательский (Custom)": []
}



def verify_dicom_directory(dicom_dir: Path) -> int:
    """
    Проверяет корректность входной папки DICOM и считает количество файлов.
    """
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"Указанный путь к DICOM не существует или не является папкой: {dicom_dir}")

    dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
    if not dicom_files:
        dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]

    num_files = len(dicom_files)
    if num_files == 0:
        raise FileNotFoundError(f"В папке {dicom_dir} не найдено DICOM-файлов.")

    logger.info(f"Найдено DICOM файлов для обработки: {num_files}")
    return num_files


def run_pipeline(
    dicom_dir_path: str,
    output_dir_path: str,
    preset_name: str,
    highres: bool = False,
    selected_organs: Optional[List[str]] = None,
    merge_mode: bool = False,
    existing_rtstruct_path: Optional[str] = None,
    step_callback: Optional[Callable[[str], None]] = None
) -> None:
    """
    Основной пайплайн выполнения автооконтурирования органов риска на КТ.
    """
    start_time = time.time()
    dicom_dir = Path(dicom_dir_path).resolve()
    output_dir = Path(output_dir_path).resolve()
    
    # Инициализация временных путей
    temp_dir = output_dir / "temp_autocontour_workspace"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_ct_path = temp_dir / "temp_ct_volume.nii.gz"
    segmentation_dir = temp_dir / "temp_masks"
    
    try:
        # Проверка DICOM-файлов
        verify_dicom_directory(dicom_dir)
        
        # Считывание PatientID из первого DICOM-файла для динамического именования
        patient_id = "Unknown"
        try:
            dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
            if not dicom_files:
                dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
            if dicom_files:
                import pydicom
                ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
                patient_id = getattr(ds, "PatientID", "Unknown")
                logger.info(f"Успешно считан PatientID из DICOM: {patient_id}")
        except Exception as de:
            logger.debug(f"Не удалось считать PatientID из DICOM: {de}")

        
        # ----------------------------------------------------------------------
        # Шаг 1: Конвертация DICOM -> NIfTI
        # ----------------------------------------------------------------------
        if step_callback:
            step_callback("Шаг 1 из 5: Конвертация DICOM в NIfTI 3D объем...")
        logger.info("--- Шаг 1 из 5: Конвертация DICOM в 3D NIfTI объем ---")
        
        settings.disable_validate_slice_increment()
        settings.disable_validate_orthogonal()
        settings.disable_validate_orientation()
        
        step_start = time.time()
        logger.info(f"Сборка 3D-тома NIfTI из {dicom_dir}... Это может занять некоторое время.")
        
        dicom2nifti.dicom_series_to_nifti(str(dicom_dir), str(nifti_ct_path), reorient_nifti=False)
        
        if not nifti_ct_path.exists():
            raise RuntimeError("Не удалось создать временный NIfTI-файл КТ.")
            
        logger.info(f"Шаг 1 успешно завершен за {time.time() - step_start:.2f} сек.")
        logger.info(f"Временный NIfTI сохранен: {nifti_ct_path} ({nifti_ct_path.stat().st_size / (1024*1024):.2f} МБ)")

        # ----------------------------------------------------------------------
        # Шаг 2: ИИ-сегментация (TotalSegmentator на CPU с оптимизацией)
        # ----------------------------------------------------------------------
        if step_callback:
            step_callback("Шаг 2 из 5: Сегментация органов нейросетью TotalSegmentator...")
        logger.info("--- Шаг 2 из 5: ИИ-сегментация с помощью TotalSegmentator ---")
        
        step_start = time.time()
        if highres:
            logger.warning(
                "ВНИМАНИЕ: Запуск сегментации принудительно на CPU в ВЫСОКОМ качестве (fast=False, 1.5 мм). "
                "Это обеспечит плавные контуры органов. Процесс на CPU может занять 5-10 минут. Пожалуйста, подождите!"
            )
        else:
            logger.warning(
                "ВНИМАНИЕ: Запуск сегментации принудительно на CPU в БЫСТРОМ режиме (fast=True, 3 мм). "
                "Для клинического качества и плавных краев запустите скрипт с флагом повышенного качества."
            )
        
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем выбранные органы
        if selected_organs is not None:
            target_organs = selected_organs
            logger.info(f"ИИ сегментирует только выбранные OAR: {target_organs}")
        else:
            target_organs = PRESETS.get(preset_name)
            if target_organs:
                logger.info(f"ИИ сегментирует только выбранные OAR из пресета '{preset_name}': {target_organs}")
            else:
                logger.warning(f"Пресет '{preset_name}' не найден. Будут экспортированы все найденные OAR.")
                target_organs = None
        
        # Запуск TotalSegmentator через внешний процесс subprocess для предотвращения крашей на Windows
        import subprocess
        
        # Находим путь к исполняемому файлу TotalSegmentator в текущем виртуальном окружении
        exe_dir = Path(sys.executable).parent
        totalseg_exe = exe_dir / "TotalSegmentator.exe"
        if not totalseg_exe.exists():
            totalseg_exe = exe_dir / "TotalSegmentator"
            
        if not totalseg_exe.exists():
            totalseg_exe = Path("TotalSegmentator")
            
        # Автоматическое определение лучшего вычислительного устройства (GPU CUDA или CPU)
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "gpu"
                logger.info("Обнаружена видеокарта с поддержкой CUDA. Сегментация будет запущена на GPU!")
            else:
                logger.info("Видеокарта с поддержкой CUDA не обнаружена. Сегментация будет запущена на CPU.")
        except Exception as e:
            logger.debug(f"Не удалось проверить доступность CUDA через PyTorch: {e}. Запуск на CPU.")

        cmd = [
            str(totalseg_exe),
            "-i", str(nifti_ct_path),
            "-o", str(segmentation_dir),
            "--device", device
        ]
        
        if not highres:
            cmd.append("--fast")
            
        if target_organs:
            cmd.append("--roi_subset")
            cmd.extend(target_organs)
            
        logger.info(f"Запуск внешнего процесса TotalSegmentator: {' '.join(cmd)}")
        
        # Скрываем черное окно консоли на Windows при запуске subprocess
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
        # Запуск процесса с перехватом stdout и stderr
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=startupinfo
        )
        
        # Чтение вывода в реальном времени
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean_line = line.strip()
                if clean_line:
                    logger.info(f"[TotalSegmentator]: {clean_line}")
                    
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Процесс TotalSegmentator завершился с кодом ошибки {return_code}")
            
        logger.info(f"Шаг 2 успешно завершен за {time.time() - step_start:.2f} сек.")

        # ----------------------------------------------------------------------
        # Шаг 3: Очистка временных файлов
        # ----------------------------------------------------------------------
        if step_callback:
            step_callback("Шаг 3 из 5: Удаление временных файлов и очистка ОЗУ...")
        logger.info("--- Шаг 3 из 5: Удаление временного NIfTI КТ и очистка ОЗУ ---")
        step_start = time.time()
        
        if nifti_ct_path.exists():
            nifti_ct_path.unlink()
            
        gc.collect()
        logger.info(f"Шаг 3 успешно завершен за {time.time() - step_start:.2f} сек. Память очищена.")

        # ----------------------------------------------------------------------
        # Шаг 4: Сборка масок в DICOM RTSTRUCT
        # ----------------------------------------------------------------------
        if step_callback:
            step_callback("Шаг 4 из 5: Сборка RTSTRUCT и привязка к геометрии DICOM...")
        logger.info("--- Шаг 4 из 5: Сборка RTSTRUCT и привязка к геометрии DICOM ---")
        
        step_start = time.time()
        
        # Получаем список масок из папки сегментации
        mask_files = list(segmentation_dir.glob("*.nii.gz"))
        if not mask_files:
            raise RuntimeError("Не найдено масок органов после сегментации.")
            
        detected_organs = sorted([f.name.replace(".nii.gz", "") for f in mask_files])
        logger.info(f"Обнаружено сегментированных масок органов: {len(mask_files)}")
        logger.info(f"Список определенных ИИ органов на КТ: {detected_organs}")
        
        # Инициализируем или загружаем существующий RTSTRUCT
        existing_rois = []
        if merge_mode and existing_rtstruct_path:
            logger.info(f"Загрузка существующего RTSTRUCT для слияния: {existing_rtstruct_path}")
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(dicom_dir),
                rt_struct_path=str(existing_rtstruct_path)
            )
            existing_rois = rtstruct.get_roi_names()
            logger.info(f"Существующие структуры врача в файле: {existing_rois}")
        else:
            logger.info("Инициализация нового RTSTRUCT считыванием оригинальной геометрии DICOM серии...")
            rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_dir))
        
        added_count = 0
        for mask_file in mask_files:
            organ_name = mask_file.name.replace(".nii.gz", "")
            
            # Фильтруем по списку целевых органов
            if target_organs and organ_name not in target_organs:
                continue
                
            logger.info(f"Обработка органа: {organ_name}...")
            
            nii_mask = nib.load(str(mask_file))
            mask_data = nii_mask.get_fdata() > 0.5
            
            # Транспонируем (X, Y, Z) к NumPy (Y, X, Z) [Rows, Cols, Slices]
            mask_data_transposed = np.transpose(mask_data, (1, 0, 2))
            mask_bool = mask_data_transposed.astype(bool)
            
            if not np.any(mask_bool):
                logger.info(f"Пропуск пустого органа: {organ_name} (отсутствует в КТ объеме)")
                continue
                
            color = ORGAN_COLORS.get(organ_name, [128, 128, 128])
            pretty_name = organ_name.replace("_", " ").title()
            
            # Умное слияние с контурами врача
            if pretty_name in existing_rois:
                pretty_name = f"{pretty_name} (AI)"
                logger.warning(f"Орган '{organ_name}' уже размечен врачом. ИИ-контур добавлен как '{pretty_name}'")
            
            rtstruct.add_roi(
                mask=mask_bool,
                color=color,
                name=pretty_name
            )
            added_count += 1
            logger.info(f"Успешно добавлен ROI '{pretty_name}' (цвет: {color})")
            
        if added_count == 0:
            raise RuntimeError("В RTSTRUCT не было добавлено ни одного OAR. Проверьте соответствие КТ-области выбранным органам.")
            
        # ----------------------------------------------------------------------
        # Шаг 5: Сохранение итогового файла
        # ----------------------------------------------------------------------
        if step_callback:
            step_callback("Шаг 5 из 5: Запись итогового DICOM RTSTRUCT...")
        logger.info("--- Шаг 5 из 5: Запись итогового DICOM RTSTRUCT ---")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Очистка PatientID для безопасного имени файла
        clean_patient_id = "".join([c for c in str(patient_id) if c.isalnum() or c in ("_", "-")]).strip()
        if not clean_patient_id:
            clean_patient_id = "Unknown"

        if merge_mode and existing_rtstruct_path:
            orig_name = Path(existing_rtstruct_path).parent.name if Path(existing_rtstruct_path).stem == "rtstruct" else Path(existing_rtstruct_path).stem
            # Если имя rtstruct, лучше взять имя родительской папки для уникальности, иначе stem
            if orig_name.lower() == "rtstruct":
                orig_name = Path(existing_rtstruct_path).parent.name
            rtstruct_filename = f"RTSTRUCT_{orig_name}_merged.dcm"
        else:
            rtstruct_filename = f"RTSTRUCT_AI_{clean_patient_id}.dcm"

        rtstruct_file_path = output_dir / rtstruct_filename
        
        rtstruct.save(str(rtstruct_file_path))
        logger.info(f"Шаг 5 успешно завершен за {time.time() - step_start:.2f} сек.")
        if merge_mode and existing_rtstruct_path:
            logger.info("Слияние успешно завершено! Исходный файл врача во входной папке КТ не изменен.")
            logger.info(f"Результат слияния успешно записан в выходную папку: {rtstruct_file_path}")
        else:
            logger.info(f"Итоговый файл RTSTRUCT успешно записан: {rtstruct_file_path}")
        
    except Exception as e:
        logger.error(f"Произошел критический сбой во время выполнения пайплайна: {e}", exc_info=True)
        logger.warning(f"ВНИМАНИЕ: Временная папка с данными сохранена для отладки: {temp_dir}")
        raise e
        
    else:
        logger.info("Очистка временных папок и файлов...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    finally:
        logger.info(f"Пайплайн завершен. Общее время работы: {time.time() - start_time:.2f} сек.")


# ==============================================================================
# Раздел графического интерфейса PyQt6
# ==============================================================================

if PYQT_AVAILABLE:
    class LogSignaler(QObject):
        """Вспомогательный класс сигналов для потокобезопасного вывода логов."""
        log_signal = pyqtSignal(str, str)

    class StreamToSignaler:
        """Перенаправляет текстовые потоки в сигналы PyQt6."""
        def __init__(self, signaler: LogSignaler, level: str = "INFO"):
            self.signaler = signaler
            self.level = level
            self.buffer = ""

        def write(self, message):
            if message:
                self.buffer += message
                if "\n" in self.buffer:
                    lines = self.buffer.split("\n")
                    self.buffer = lines[-1]
                    for line in lines[:-1]:
                        if line.strip():
                            color = "#ecf0f1"
                            if "ERROR" in line or "Exception" in line or self.level == "ERROR":
                                color = "#ff6b6b"  # Красный
                            elif "WARNING" in line:
                                color = "#f1c40f"  # Желтый
                            elif "Шаг" in line or "---" in line:
                                color = "#007acc"  # Синий
                            self.signaler.log_signal.emit(line, color)

        def flush(self):
            if self.buffer.strip():
                color = "#ecf0f1"
                if self.level == "ERROR":
                    color = "#ff6b6b"
                self.signaler.log_signal.emit(self.buffer, color)
                self.buffer = ""

        def isatty(self):
            return False

    class QTextEditLogHandler(logging.Handler):
        """Обработчик logging для перенаправления логов в QTextEdit."""
        def __init__(self, signaler: LogSignaler):
            super().__init__()
            self.signaler = signaler
            self.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))

        def emit(self, record):
            try:
                msg = self.format(record)
                if record.levelno == logging.ERROR:
                    color = "#ff6b6b"
                elif record.levelno == logging.WARNING:
                    color = "#f1c40f"
                else:
                    color = "#a0a0a2"
                self.signaler.log_signal.emit(msg, color)
            except Exception:
                self.handleError(record)

    class SegmentationWorker(QThread):
        """Поток для вычислений сегментации TotalSegmentator, чтобы GUI не зависал."""
        finished_signal = pyqtSignal(bool, str)
        step_signal = pyqtSignal(str)

        def __init__(
            self,
            dicom_dir: str,
            output_dir: str,
            preset_name: str,
            highres: bool,
            selected_organs: List[str],
            merge_mode: bool,
            existing_rtstruct_path: Optional[str]
        ):
            super().__init__()
            self.dicom_dir = dicom_dir
            self.output_dir = output_dir
            self.preset_name = preset_name
            self.highres = highres
            self.selected_organs = selected_organs
            self.merge_mode = merge_mode
            self.existing_rtstruct_path = existing_rtstruct_path

        def run(self):
            try:
                def callback(step_text: str):
                    self.step_signal.emit(step_text)

                run_pipeline(
                    dicom_dir_path=self.dicom_dir,
                    output_dir_path=self.output_dir,
                    preset_name=self.preset_name,
                    highres=self.highres,
                    selected_organs=self.selected_organs,
                    merge_mode=self.merge_mode,
                    existing_rtstruct_path=self.existing_rtstruct_path,
                    step_callback=callback
                )
                self.finished_signal.emit(True, "Автооконтурирование успешно завершено!")
            except Exception as e:
                self.finished_signal.emit(False, str(e))

    # Стилизация премиальной темной темы QSS
    DARK_QSS = """
    QWidget {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 13px;
    }

    QFrame#card {
        background-color: #242424;
        border: 1px solid #333333;
        border-radius: 8px;
    }

    QFrame#statusCard {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
    }

    QLabel {
        color: #b0b0b0;
        background-color: transparent;
    }

    QLabel#titleLabel {
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
    }

    QLabel#subtitleLabel {
        color: #007acc;
        font-size: 12px;
        font-weight: 600;
    }

    QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        padding: 6px 10px;
        color: #ffffff;
    }

    QLineEdit:focus {
        border: 1px solid #007acc;
    }

    QPushButton {
        background-color: #333333;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 6px 12px;
        color: #ffffff;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #444444;
        border: 1px solid #555555;
    }

    QPushButton:pressed {
        background-color: #222222;
    }

    QPushButton#btnBrowse {
        background-color: #2d2d2d;
    }

    QPushButton#btnBrowse:hover {
        background-color: #3d3d3d;
    }

    QPushButton#btnRun {
        background-color: #007acc;
        border: 1px solid #007acc;
        font-size: 14px;
        font-weight: bold;
        padding: 12px;
        border-radius: 6px;
        color: #ffffff;
    }

    QPushButton#btnRun:hover {
        background-color: #0098ff;
        border: 1px solid #0098ff;
    }

    QPushButton#btnRun:disabled {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
        color: #888888;
    }

    QPushButton#btnAction {
        background-color: #2b2b2b;
        border: 1px solid #3d3d3d;
        font-size: 12px;
        padding: 6px 12px;
        border-radius: 4px;
        color: #e0e0e0;
    }

    QPushButton#btnAction:hover {
        background-color: #3d3d3d;
        border: 1px solid #007acc;
        color: #ffffff;
    }


    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        padding: 5px 10px;
        color: #ffffff;
    }

    QComboBox::drop-down {
        border: 0px;
    }

    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        border: 1px solid #3c3c3c;
        selection-background-color: #007acc;
        selection-color: #ffffff;
    }

    QListWidget {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        padding: 5px;
    }

    QListWidget::item {
        padding: 4px;
    }

    QListWidget::item:hover {
        background-color: #2d2d2d;
        border-radius: 4px;
    }

    QRadioButton {
        spacing: 8px;
        color: #d0d0d0;
    }

    QRadioButton::disabled {
        color: #666666;
    }

    QCheckBox {
        spacing: 8px;
        color: #d0d0d0;
    }

    QProgressBar {
        border: 1px solid #333333;
        border-radius: 4px;
        text-align: center;
        background-color: #1e1e1e;
        height: 18px;
        color: #ffffff;
        font-weight: bold;
    }

    QProgressBar::chunk {
        background-color: #007acc;
        border-radius: 3px;
    }

    QTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #2d2d2d;
        border-radius: 6px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 12px;
        padding: 8px;
        color: #ecf0f1;
    }

    QScrollBar:vertical {
        border: 0px;
        background: #1a1a1a;
        width: 10px;
        margin: 0px;
    }

    QScrollBar::handle:vertical {
        background: #444444;
        min-height: 20px;
        border-radius: 5px;
    }

    QScrollBar::handle:vertical:hover {
        background: #555555;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
    }

    QPushButton#btnHelp {
        background-color: #2b2b2b;
        border: 1px solid #3d3d3d;
        color: #007acc;
        padding: 5px 12px;
        font-size: 13px;
        font-weight: bold;
        border-radius: 4px;
    }

    QPushButton#btnHelp:hover {
        background-color: #333333;
        border: 1px solid #007acc;
        color: #0098ff;
    }
    """

    class MainWindow(QMainWindow):
        """Главное окно графического интерфейса приложения."""
        def __init__(self):
            super().__init__()
            self.setWindowTitle("AI Contour - Автооконтурирование КТ органов риска")
            self.setMinimumSize(920, 720)
            self.existing_rtstruct_path = None
            self.is_updating_presets = False
            self.worker = None
            self.settings = QSettings("AIContourCorp", "AIContour")

            # Настройка перенаправления логов в реальном времени
            self.log_signaler = LogSignaler()
            self.log_signaler.log_signal.connect(self.append_log)
            self.log_handler = QTextEditLogHandler(self.log_signaler)
            logger.addHandler(self.log_handler)

            # Таймер активности (спиннер + пульсация цвета)
            self.activity_timer = QTimer(self)
            self.activity_timer.setInterval(120)
            self.activity_timer.timeout.connect(self.update_activity_animation)
            self.spinner_index = 0
            self.pulse_tick = 0
            self.current_step_base_text = "Ожидание запуска..."
            self.SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

            self.init_ui()
            self.load_settings()

        def init_ui(self):
            self.setStyleSheet(DARK_QSS)

            # Главный виджет
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QVBoxLayout(main_widget)
            main_layout.setContentsMargins(15, 15, 15, 15)
            main_layout.setSpacing(10)

            # Определение GPU/CPU для подзаголовка
            device_str = "CPU"
            try:
                import torch
                if torch.cuda.is_available():
                    device_str = "CUDA GPU"
            except Exception:
                pass

            # Шапка
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 5)

            title_layout = QVBoxLayout()
            title_layout.setSpacing(2)
            title = QLabel("AI Contour")
            title.setObjectName("titleLabel")
            subtitle = QLabel(f"Автоматическое сегментирование органов риска на КТ (TotalSegmentator {device_str})")
            subtitle.setObjectName("subtitleLabel")
            title_layout.addWidget(title)
            title_layout.addWidget(subtitle)

            btn_help = QPushButton("Справка и дисклеймер 📖")
            btn_help.setObjectName("btnHelp")
            btn_help.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_help.clicked.connect(self.show_help)

            header_layout.addLayout(title_layout)
            header_layout.addStretch()
            header_layout.addWidget(btn_help)
            main_layout.addWidget(header_widget)

            # Сплиттер
            splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(splitter, 1)

            # --- ЛЕВАЯ КОЛОНКА (Настройки и параметры) ---
            left_card = QFrame()
            left_card.setObjectName("card")
            left_card.setMinimumWidth(380)
            left_card.setMaximumWidth(460)
            left_layout = QVBoxLayout(left_card)
            left_layout.setSpacing(12)

            # Выбор КТ DICOM
            input_label = QLabel("Папка с КТ-снимками DICOM:")
            input_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.input_edit = QLineEdit()
            self.input_edit.setPlaceholderText("Выберите папку с DICOM файлами снимков...")
            self.input_edit.textChanged.connect(self.check_for_rtstruct)
            btn_input = QPushButton("Обзор...")
            btn_input.setObjectName("btnBrowse")
            btn_input.clicked.connect(self.select_input_dir)

            input_box = QHBoxLayout()
            input_box.addWidget(self.input_edit)
            input_box.addWidget(btn_input)
            left_layout.addWidget(input_label)
            left_layout.addLayout(input_box)

            # Выбор директории вывода
            output_label = QLabel("Папка сохранения результатов:")
            output_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.output_edit = QLineEdit()
            self.output_edit.setPlaceholderText("Выберите выходную папку для RTSTRUCT...")
            btn_output = QPushButton("Обзор...")
            btn_output.setObjectName("btnBrowse")
            btn_output.clicked.connect(self.select_output_dir)

            output_box = QHBoxLayout()
            output_box.addWidget(self.output_edit)
            output_box.addWidget(btn_output)
            left_layout.addWidget(output_label)
            left_layout.addLayout(output_box)

            # Под-карточка статуса RTSTRUCT
            status_frame = QFrame()
            status_frame.setObjectName("statusCard")
            status_layout = QVBoxLayout(status_frame)
            status_layout.setSpacing(8)

            status_title = QLabel("Работа с существующими контурами:")
            status_title.setStyleSheet("font-weight: bold; color: #b0b0b0;")
            self.status_rtstruct_label = QLabel("Статус: Путь не выбран")
            self.status_rtstruct_label.setStyleSheet("color: #888888;")
            self.status_rtstruct_label.setWordWrap(True)

            self.radio_merge = QRadioButton("Дописать ИИ-контуры в существующий файл врача (Merge)")
            self.radio_new = QRadioButton("Создать новый файл отдельно (Сохранить оригинал)")
            self.radio_merge.setEnabled(False)
            self.radio_new.setEnabled(False)
            self.radio_new.setChecked(True)

            self.radio_group = QButtonGroup()
            self.radio_group.addButton(self.radio_merge)
            self.radio_group.addButton(self.radio_new)

            status_layout.addWidget(status_title)
            status_layout.addWidget(self.status_rtstruct_label)
            status_layout.addWidget(self.radio_merge)
            status_layout.addWidget(self.radio_new)
            left_layout.addWidget(status_frame)

            # Выбор пресета
            preset_label = QLabel("Выбор пресета органов (OAR):")
            preset_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.preset_combo = QComboBox()
            self.preset_combo.addItems(list(PRESETS_MAP.keys()))
            self.preset_combo.currentTextChanged.connect(self.on_preset_changed)

            left_layout.addWidget(preset_label)
            left_layout.addWidget(self.preset_combo)

            # Кнопки быстрого выделения органов риска
            selection_layout = QHBoxLayout()
            btn_select_all = QPushButton("Выбрать все")
            btn_select_all.setObjectName("btnAction")
            btn_select_all.setToolTip("Отметить абсолютно все органы риска во всех группах")
            btn_select_all.clicked.connect(self.select_all_organs)
            
            btn_deselect_all = QPushButton("Снять все")
            btn_deselect_all.setObjectName("btnAction")
            btn_deselect_all.setToolTip("Снять выделение со всех органов риска")
            btn_deselect_all.clicked.connect(self.deselect_all_organs)
            
            selection_layout.addWidget(btn_select_all)
            selection_layout.addWidget(btn_deselect_all)
            left_layout.addLayout(selection_layout)

            # Список OAR с чек-боксами
            organs_header = QLabel("Органы для автооконтурирования:")
            organs_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.organs_list = QListWidget()
            self.organs_list.itemChanged.connect(self.on_organ_item_changed)

            # Заполнение списка с группировкой по анатомическим областям (по протоколам QUANTEC/TG-263)
            ORGAN_GROUPS = {
                "--- ГОЛОВА И ШЕЯ ---": [
                    "brain", "spinal_cord", "thyroid_gland", "skull", "trachea", "esophagus",
                    "common_carotid_artery_left", "common_carotid_artery_right"
                ],
                "--- ГРУДНАЯ КЛЕТКА ---": [
                    "heart", "lung_left", "lung_right", "trachea", "esophagus", "aorta", "pulmonary_artery",
                    "superior_vena_cava", "sternum", "clavicula_left", "clavicula_right"
                ],
                "--- БРЮШНАЯ ПОЛОСТЬ ---": [
                    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "inferior_vena_cava", "pancreas", "duodenum", "adrenal_gland_left", "adrenal_gland_right", "portal_vein_and_splenic_vein"
                ],
                "--- МАЛЫЙ ТАЗ ---": [
                    "urinary_bladder", "prostate", "rectum", "colon", "small_bowel", "femur_left", "femur_right", "hip_left", "hip_right", "sacrum", "iliac_artery_left", "iliac_artery_right"
                ]
            }

            for group_title, organs in ORGAN_GROUPS.items():
                # Элемент-заголовок группы
                header_item = QListWidgetItem(group_title)
                header_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Невыбираемый, без чекбокса
                header_item.setData(Qt.ItemDataRole.UserRole, "header")
                
                # Стилизация заголовка
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                header_item.setForeground(QBrush(QColor("#007acc")))  # Фирменный синий цвет
                header_item.setBackground(QBrush(QColor("#242424")))  # Чуть светлее фона списка для контраста
                
                self.organs_list.addItem(header_item)

                # Добавление органов группы
                for org in organs:
                    ru_name = ORGAN_RU_NAMES.get(org, org)
                    item = QListWidgetItem(f"   {ru_name}")  # Отступ для визуального выделения иерархии
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    self.organs_list.addItem(item)

            left_layout.addWidget(organs_header)
            left_layout.addWidget(self.organs_list)

            # Точность
            self.highres_check = QCheckBox("Повышенная точность (1.5 мм, медленный расчет на CPU)")
            self.highres_check.setToolTip("Использует высокое разрешение КТ. Занимает значительно больше времени и ОЗУ.")
            left_layout.addWidget(self.highres_check)

            splitter.addWidget(left_card)

            # --- ПРАВАЯ КОЛОНКА (Терминал логов и управление) ---
            right_card = QFrame()
            right_card.setObjectName("card")
            right_layout = QVBoxLayout(right_card)
            right_layout.setSpacing(12)

            logs_header = QLabel("Лог выполнения работы ИИ в реальном времени:")
            logs_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.log_edit = QTextEdit()
            self.log_edit.setReadOnly(True)
            self.log_edit.setPlaceholderText("Здесь будет отображаться ход выполнения автооконтурирования...")

            progress_header = QLabel("Индикатор прогресса:")
            progress_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.status_step_label = QLabel("Текущий шаг: Ожидание запуска...")
            self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setFormat("%p%")

            self.btn_run = QPushButton("ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ")
            self.btn_run.setObjectName("btnRun")
            self.btn_run.clicked.connect(self.start_segmentation)

            right_layout.addWidget(logs_header)
            right_layout.addWidget(self.log_edit)
            right_layout.addWidget(progress_header)
            right_layout.addWidget(self.status_step_label)
            right_layout.addWidget(self.progress_bar)
            right_layout.addWidget(self.btn_run)

            splitter.addWidget(right_card)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)

            # Подключаем сохранение настроек при смене флага точности
            self.highres_check.stateChanged.connect(self.save_settings)
            splitter.setSizes([400, 520])

        def load_settings(self):
            """Загружает сохраненное состояние интерфейса."""
            # Блокируем сигналы, чтобы избежать автовызовов и лишних циклов обновлений при инициализации
            self.preset_combo.blockSignals(True)
            self.organs_list.blockSignals(True)
            self.is_updating_presets = True
            
            try:
                input_dir = self.settings.value("input_dir", "")
                output_dir = self.settings.value("output_dir", "")
                if input_dir:
                    self.input_edit.setText(input_dir)
                if output_dir:
                    self.output_edit.setText(output_dir)
                
                # Загружаем пресет
                preset = self.settings.value("preset", "Брюшная полость (Abdomen)")
                self.preset_combo.setCurrentText(preset)
                
                # Загружаем точность
                highres = self.settings.value("highres", False, type=bool)
                self.highres_check.setChecked(highres)
                
                # Загружаем отмеченные органы
                checked_organs = self.settings.value("checked_organs", None)
                
                if checked_organs is not None:
                    # Если есть сохраненный список органов
                    if not isinstance(checked_organs, list):
                        checked_organs = [checked_organs] # На случай, если QSettings вернул одиночную строку
                    
                    for i in range(self.organs_list.count()):
                        item = self.organs_list.item(i)
                        organ_name = item.data(Qt.ItemDataRole.UserRole)
                        if organ_name == "header":
                            continue
                        if organ_name in checked_organs:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    # Если это первый запуск, отмечаем структуры по пресету по умолчанию
                    target_organs = PRESETS_MAP.get(preset, [])
                    for i in range(self.organs_list.count()):
                        item = self.organs_list.item(i)
                        organ_name = item.data(Qt.ItemDataRole.UserRole)
                        if organ_name == "header":
                            continue
                        if organ_name in target_organs:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)
            finally:
                self.is_updating_presets = False
                self.organs_list.blockSignals(False)
                self.preset_combo.blockSignals(False)

        def save_settings(self):
            """Сохраняет состояние интерфейса в реестр / конфиг."""
            self.settings.setValue("input_dir", self.input_edit.text().strip())
            self.settings.setValue("output_dir", self.output_edit.text().strip())
            self.settings.setValue("preset", self.preset_combo.currentText())
            self.settings.setValue("highres", self.highres_check.isChecked())
            
            # Собираем список всех выбранных органов
            checked_organs = []
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                organ_name = item.data(Qt.ItemDataRole.UserRole)
                if organ_name == "header":
                    continue
                if item.checkState() == Qt.CheckState.Checked:
                    if organ_name not in checked_organs:
                        checked_organs.append(organ_name)
            self.settings.setValue("checked_organs", checked_organs)

        def select_input_dir(self):
            dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку с КТ-снимками DICOM")
            if dir_path:
                self.input_edit.setText(dir_path)
                if not self.output_edit.text():
                    self.output_edit.setText(os.path.join(dir_path, "output"))
                self.save_settings()

        def select_output_dir(self):
            dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку сохранения результатов")
            if dir_path:
                self.output_edit.setText(dir_path)
                self.save_settings()

        def check_for_rtstruct(self, directory: str):
            """Автоматически сканирует папку КТ на наличие существующего RTSTRUCT файла."""
            self.existing_rtstruct_path = None
            if not directory or not os.path.isdir(directory):
                self.status_rtstruct_label.setText("Статус: Путь не выбран или недействителен")
                self.status_rtstruct_label.setStyleSheet("color: #888888;")
                self.radio_merge.setEnabled(False)
                self.radio_new.setEnabled(False)
                self.radio_new.setChecked(True)
                return

            self.status_rtstruct_label.setText("Сканирование папки на наличие RTSTRUCT...")
            self.status_rtstruct_label.setStyleSheet("color: #f1c40f;")
            QApplication.processEvents()

            try:
                import pydicom
                found_file = None
                
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        try:
                            # Проверяем DICOM-заголовок без чтения пикселей
                            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                            if getattr(ds, "Modality", None) == "RTSTRUCT":
                                found_file = filepath
                                break
                        except Exception:
                            continue
                
                if found_file:
                    self.existing_rtstruct_path = found_file
                    basename = os.path.basename(found_file)
                    self.status_rtstruct_label.setText(f"Обнаружен существующий RTSTRUCT врача: {basename}")
                    self.status_rtstruct_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
                    self.radio_merge.setEnabled(True)
                    self.radio_new.setEnabled(True)
                    self.radio_merge.setChecked(True)
                else:
                    self.status_rtstruct_label.setText("Существующий RTSTRUCT врача не обнаружен (будет создан новый)")
                    self.status_rtstruct_label.setStyleSheet("color: #e74c3c;")
                    self.radio_merge.setEnabled(False)
                    self.radio_new.setEnabled(False)
                    self.radio_new.setChecked(True)
                    
            except Exception as e:
                self.status_rtstruct_label.setText(f"Ошибка при сканировании RTSTRUCT: {str(e)}")
                self.status_rtstruct_label.setStyleSheet("color: #e74c3c;")
                self.radio_merge.setEnabled(False)
                self.radio_new.setEnabled(False)
                self.radio_new.setChecked(True)

        def select_all_organs(self):
            """Отмечает все органы в списке."""
            self.is_updating_presets = True
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                organ_name = item.data(Qt.ItemDataRole.UserRole)
                if organ_name == "header":
                    continue
                item.setCheckState(Qt.CheckState.Checked)
            self.is_updating_presets = False
            
            # Обновляем комбобокс пресетов
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Все органы (All)")
            self.preset_combo.blockSignals(False)
            
            self.save_settings()

        def deselect_all_organs(self):
            """Снимает выбор со всех органов в списке."""
            self.is_updating_presets = True
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                organ_name = item.data(Qt.ItemDataRole.UserRole)
                if organ_name == "header":
                    continue
                item.setCheckState(Qt.CheckState.Unchecked)
            self.is_updating_presets = False
            
            # Обновляем комбобокс пресетов
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Пользовательский (Custom)")
            self.preset_combo.blockSignals(False)
            
            self.save_settings()

        def on_preset_changed(self, text: str):
            """Слот изменения выбранного пресета."""
            if text == "Пользовательский (Custom)":
                return
                
            self.is_updating_presets = True
            target_organs = PRESETS_MAP.get(text, [])
            
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                organ_name = item.data(Qt.ItemDataRole.UserRole)
                if organ_name == "header":
                    continue
                if organ_name in target_organs:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    
            self.is_updating_presets = False
            self.save_settings()

        def on_organ_item_changed(self, item: QListWidgetItem):
            """Слот изменения состояния чекбокса органа пользователем."""
            if self.is_updating_presets:
                return
                
            organ_name = item.data(Qt.ItemDataRole.UserRole)
            if organ_name == "header":
                return
                
            self.is_updating_presets = True
            # Синхронизируем состояние чекбоксов для одинаковых органов в других анатомических группах
            state = item.checkState()
            for i in range(self.organs_list.count()):
                itm = self.organs_list.item(i)
                if itm != item and itm.data(Qt.ItemDataRole.UserRole) == organ_name:
                    itm.setCheckState(state)
            self.is_updating_presets = False
                
            # Собираем все выбранные органы (только уникальные)
            checked_organs = []
            for i in range(self.organs_list.count()):
                itm = self.organs_list.item(i)
                org = itm.data(Qt.ItemDataRole.UserRole)
                if org == "header":
                    continue
                if itm.checkState() == Qt.CheckState.Checked:
                    if org not in checked_organs:
                        checked_organs.append(org)
                    
            # Проверяем на соответствие пресетам
            matched_preset = "Пользовательский (Custom)"
            for preset_name, preset_organs in PRESETS_MAP.items():
                if preset_name == "Пользовательский (Custom)":
                    continue
                if set(checked_organs) == set(preset_organs):
                    matched_preset = preset_name
                    break
                    
            # Блокируем сигналы комбобокса на время смены названия
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(matched_preset)
            self.preset_combo.blockSignals(False)
            self.save_settings()

        def append_log(self, message: str, color: str):
            """Потокобезопасное добавление логов в текстовое окно."""
            self.log_edit.append(f"<span style='color: {color};'>{message}</span>")
            self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

        def start_segmentation(self):
            """Запускает процесс сегментации."""
            dicom_dir = self.input_edit.text().strip()
            output_dir = self.output_edit.text().strip()
            
            if not dicom_dir or not os.path.isdir(dicom_dir):
                QMessageBox.critical(self, "Ошибка", "Укажите корректный путь к папке с КТ DICOM снимками!")
                return
                
            if not output_dir:
                QMessageBox.critical(self, "Ошибка", "Укажите выходную папку для сохранения результатов!")
                return
                
            selected_organs = []
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                org = item.data(Qt.ItemDataRole.UserRole)
                if org == "header":
                    continue
                if item.checkState() == Qt.CheckState.Checked:
                    if org not in selected_organs:
                        selected_organs.append(org)
                    
            if not selected_organs:
                QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одного органа для сегментирования!")
                return
                
            # Блокируем интерфейс
            self.set_ui_enabled(False)
            self.log_edit.clear()
            self.progress_bar.setValue(0)
            
            merge_mode = self.radio_merge.isChecked()
            preset_name = self.preset_combo.currentText()
            
            preset_key = "abdominal_oar"
            if "Thorax" in preset_name or "Грудная" in preset_name:
                preset_key = "thoracic_oar"
            elif "Pelvis" in preset_name or "Малый" in preset_name:
                preset_key = "pelvis_oar"
            elif "Head & Neck" in preset_name or "Голова" in preset_name:
                preset_key = "head_neck_oar"
            else:
                preset_key = "all"
                
            # Создаем и запускаем вычислительный поток
            self.worker = SegmentationWorker(
                dicom_dir=dicom_dir,
                output_dir=output_dir,
                preset_name=preset_key,
                highres=self.highres_check.isChecked(),
                selected_organs=selected_organs,
                merge_mode=merge_mode,
                existing_rtstruct_path=self.existing_rtstruct_path
            )
            self.worker.finished_signal.connect(self.on_segmentation_finished)
            self.worker.step_signal.connect(self.on_step_changed)
            
            # Запуск анимации активности
            self.current_step_base_text = "Подготовка пайплайна..."
            self.spinner_index = 0
            self.pulse_tick = 0
            self.activity_timer.start()
            
            self.worker.start()

        def set_ui_enabled(self, enabled: bool):
            self.input_edit.setEnabled(enabled)
            self.output_edit.setEnabled(enabled)
            self.preset_combo.setEnabled(enabled)
            self.organs_list.setEnabled(enabled)
            self.highres_check.setEnabled(enabled)
            self.radio_merge.setEnabled(enabled if self.existing_rtstruct_path else False)
            self.radio_new.setEnabled(enabled if self.existing_rtstruct_path else False)
            self.btn_run.setEnabled(enabled)
            if enabled:
                self.btn_run.setText("ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ")
            else:
                self.btn_run.setText("ВЫПОЛНЯЕТСЯ СЕГМЕНТАЦИЯ...")

        def update_activity_animation(self):
            """Обновляет анимацию вращения спиннера и плавного пульсирования цвета."""
            self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
            spinner_char = self.SPINNER_FRAMES[self.spinner_index]
            self.status_step_label.setText(f"{self.current_step_base_text} {spinner_char}")
            
            # Анимация плавного пульсирования цвета (синусоида)
            self.pulse_tick += 1
            factor = (math.sin(self.pulse_tick * 0.15) + 1.0) / 2.0
            
            # Интерполируем между #007acc (rgb 0, 122, 204) и #00e5ff (rgb 0, 229, 255)
            g = int(122 + (229 - 122) * factor)
            b = int(204 + (255 - 204) * factor)
            
            self.status_step_label.setStyleSheet(
                f"color: rgb(0, {g}, {b}); font-weight: bold; font-style: italic;"
            )

        def on_segmentation_finished(self, success: bool, message: str):
            self.set_ui_enabled(True)
            self.progress_bar.setRange(0, 100)
            
            # Останавливаем анимацию активности и сбрасываем стиль на статичный приятный синий
            self.activity_timer.stop()
            self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
            
            if success:
                self.progress_bar.setValue(100)
                self.status_step_label.setText("Текущий шаг: Готово!")
                QMessageBox.information(self, "Успех", "Автоматическое оконтурирование завершено успешно!")
            else:
                self.progress_bar.setValue(0)
                self.status_step_label.setText("Текущий шаг: Ошибка во время расчетов!")
                QMessageBox.critical(self, "Критическая ошибка", f"Произошел сбой при сегментации:\n{message}")

        def on_step_changed(self, step_text: str):
            """Слот изменения текущего текстового шага пайплайна."""
            self.current_step_base_text = step_text
            self.status_step_label.setText(f"{step_text} {self.SPINNER_FRAMES[self.spinner_index]}")
            
            # Динамическое обновление процентов на основе шагов пайплайна
            if "Шаг 1" in step_text:
                self.progress_bar.setValue(10)
            elif "Шаг 2" in step_text:
                self.progress_bar.setValue(30)
            elif "Шаг 3" in step_text:
                self.progress_bar.setValue(75)
            elif "Шаг 4" in step_text:
                self.progress_bar.setValue(85)
            elif "Шаг 5" in step_text:
                self.progress_bar.setValue(95)

        def show_help(self):
            """Открывает диалоговое окно со справкой и дисклеймером."""
            dialog = QDialog(self)
            dialog.setWindowTitle("Справка и медицинский дисклеймер")
            dialog.setMinimumSize(640, 560)
            dialog.setStyleSheet(self.styleSheet())
            
            # Вертикальный макет для диалога
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(12)
            
            # QTextBrowser для рендеринга HTML справки
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            
            html_content = """<!DOCTYPE html>
<html>
<head>
<style>
    body {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13.5px;
        line-height: 1.6;
        margin: 0;
        padding: 5px;
    }
    h1 {
        color: #ffffff;
        font-size: 19px;
        border-bottom: 2px solid #007acc;
        padding-bottom: 8px;
        margin-top: 0;
    }
    h2 {
        color: #007acc;
        font-size: 15px;
        margin-top: 18px;
        margin-bottom: 8px;
        font-weight: bold;
    }
    ul {
        margin: 0;
        padding-left: 20px;
    }
    li {
        margin-bottom: 6px;
    }
    .disclaimer-box {
        background-color: #2c1a1a;
        border: 1px solid #d32f2f;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 18px;
        margin-bottom: 5px;
    }
    .disclaimer-title {
        color: #f44336;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .highlight {
        color: #0098ff;
        font-weight: bold;
    }
    .card {
        background-color: #242424;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
</head>
<body>
    <h1>Справка по работе с AI Contour 📖</h1>
    
    <p><b>AI Contour</b> — это интеллектуальное программное обеспечение, разработанное для автоматического сегментирования критических органов риска (OAR) на КТ-изображениях DICOM с использованием искусственной нейросети <b>TotalSegmentator</b>.</p>

    <div class="card">
        <h2>Основные возможности 🚀</h2>
        <ul>
            <li><b>41 орган риска (OAR):</b> Сегментация широкого перечня структур по международным протоколам (QUANTEC, TG-263), включая структуры головы и шеи, грудной клетки, брюшной полости и малого таза.</li>
            <li><b>Интеллектуальное GPU-ускорение:</b> Программа автоматически определяет наличие графического ускорителя Nvidia с поддержкой CUDA. На GPU сегментация занимает всего <span class="highlight">15–20 секунд</span> (вместо 5–10 минут на CPU). При отсутствии GPU безопасно используется CPU-режим.</li>
            <li><b>Анатомические пресеты:</b> Возможность мгновенного выбора органов по группам (Голова и Шея, Грудная клетка, Брюшная полость, Малый таз, Все органы) или гибкой ручной настройки (Пользовательский).</li>
            <li><b>Умное объединение (Merge):</b> Программа может записать сгенерированные ИИ-контуры прямо в существующий файл разметки врача (RTSTRUCT) без удаления или повреждения его собственных контуров, либо сохранить результаты в новый файл.</li>
            <li><b>Полное сохранение состояния:</b> Все выбранные чекбоксы, пресеты, пути к папкам и настройки точности автоматически сохраняются и восстанавливаются при следующем запуске.</li>
        </ul>
    </div>

    <div class="card">
        <h2>Технические ограничения ⚠️</h2>
        <ul>
            <li><b>Требования к КТ-снимкам:</b> КТ-исследование должно быть представлено в виде папки с валидными DICOM-файлами (без пропущенных срезов и без артефактов реконструкции).</li>
            <li><b>Высокое разрешение (Highres):</b> Режим высокой точности обеспечивает максимальную детализацию контуров органов, но требует больше времени для расчетов и большего объема RAM (рекомендуется от 16 ГБ).</li>
            <li><b>Требования для GPU-режима:</b> Требуется дискретная видеокарта Nvidia (архитектура Pascal и новее), установленные драйверы CUDA и PyTorch с поддержкой CUDA в виртуальном окружении.</li>
        </ul>
    </div>

    <div class="disclaimer-box">
        <div class="disclaimer-title">⚠️ ВАЖНЫЙ МЕДИЦИНСКИЙ ДИСКЛЕЙМЕР</div>
        <p style="margin: 0; font-size: 12.5px; color: #e0b0b0;">
            Данное программное обеспечение предоставляется исключительно для научных и исследовательских целей (<b>Research Use Only</b>). <br><br>
            Автоматическая разметка, сгенерированная искусственным интеллектом, <b>не является окончательной клинической разметкой</b>. Она <b>не должна напрямую использоваться</b> для планирования лучевой терапии, хирургических вмешательств или других клинических манипуляций без обязательной проверки. <br><br>
            Любая импортированная разметка <b>подлежит обязательному ручному контролю, валидации и коррекции</b> сертифицированным медицинским физиком или радиационным онкологом в клинической системе планирования (TPS) перед облучением пациента. Разработчики не несут ответственности за любые клинические решения, принятые на основе работы ПО.
        </p>
    </div>
</body>
</html>"""
            
            browser.setHtml(html_content)
            layout.addWidget(browser, 1)
            
            # Кнопка закрытия
            btn_close = QPushButton("Ясно, закрыть")
            btn_close.setObjectName("btnAction")
            btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_close.clicked.connect(dialog.accept)
            
            # Разместим кнопку по центру
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_layout.addWidget(btn_close)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)
            
            dialog.exec()

        def closeEvent(self, event):
            event.accept()


# ==============================================================================
# Исполняемый блок программы
# ==============================================================================

if __name__ == "__main__":
    # Если переданы аргументы командной строки, запускаем в режиме CLI
    if len(sys.argv) > 1:
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
        parser.add_argument(
            "-hr", "--highres",
            action="store_true",
            help="Запустить сегментацию в высоком качестве (1.5 мм разрешение вместо 3 мм)."
        )
        
        args = parser.parse_args()
        run_pipeline(args.input, args.output, args.preset, args.highres)
    else:
        # Режим GUI
        if not PYQT_AVAILABLE:
            print("Ошибка: Для запуска GUI необходима библиотека PyQt6.")
            print("Установите ее с помощью команды: pip install PyQt6")
            sys.exit(1)
            
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
