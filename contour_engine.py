#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Модуль contour_engine.py: Изолированный сервисный движок автооконтурирования
================================================================================
Этот модуль содержит всю тяжелую вычислительную логику:
1. Конвертация DICOM в NIfTI (dicom2nifti).
2. Запуск ИИ-сегментации TotalSegmentator с поддержкой GPU/CPU и режимов точности.
3. 3D постобработка бинарных масок (удаление артефактов, сглаживание Гаусса).
4. Запись результатов в DICOM RTSTRUCT через rt-utils.
5. Управление динамическими пресетами, цветами и локализацией через presets.json.

Полностью независим от графического интерфейса PyQt6.
================================================================================
"""

import os
import sys
import gc
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti
import dicom2nifti.settings as d2n_settings
from rt_utils import RTStructBuilder
from scipy.ndimage import label, gaussian_filter

# Настройка локального логера движка
logger = logging.getLogger("ContourEngine")

# Дефолтные настройки для автогенерации presets.json при его отсутствии
DEFAULT_PRESETS_DATA = {
    "presets": {
        "Голова и шея (Head & Neck)": [
            "brain", "spinal_cord", "thyroid_gland", "skull", "trachea", "esophagus",
            "common_carotid_artery_left", "common_carotid_artery_right"
        ],
        "Грудная клетка (Thorax)": [
            "heart", "lung_left", "lung_right", "trachea", "esophagus", "aorta", "pulmonary_artery",
            "superior_vena_cava", "sternum", "clavicula_left", "clavicula_right"
        ],
        "Брюшная полость (Abdomen)": [
            "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "aorta",
            "inferior_vena_cava", "urinary_bladder", "heart", "pancreas", "duodenum",
            "adrenal_gland_left", "adrenal_gland_right", "portal_vein_and_splenic_vein"
        ],
        "Малый таз (Pelvis)": [
            "urinary_bladder", "prostate", "rectum", "colon", "small_bowel", "femur_left", "femur_right",
            "hip_left", "hip_right", "sacrum", "iliac_artery_left", "iliac_artery_right"
        ],
        "Брахитерапия (Brachytherapy)": [
            "urinary_bladder", "small_bowel", "colon"
        ]
    },
    "colors": {
        "spleen": [156, 39, 176],
        "kidney_right": [3, 169, 244],
        "kidney_left": [33, 150, 243],
        "gallbladder": [76, 175, 80],
        "liver": [139, 195, 74],
        "stomach": [255, 152, 0],
        "aorta": [244, 67, 54],
        "inferior_vena_cava": [63, 81, 181],
        "urinary_bladder": [255, 235, 59],
        "heart": [233, 30, 99],
        "lung_left": [0, 150, 136],
        "lung_right": [0, 188, 212],
        "trachea": [121, 85, 72],
        "esophagus": [158, 158, 158],
        "pancreas": [255, 193, 7],
        "duodenum": [173, 20, 87],
        "adrenal_gland_left": [255, 87, 34],
        "adrenal_gland_right": [255, 112, 67],
        "pulmonary_artery": [0, 150, 255],
        "small_bowel": [103, 58, 183],
        "prostate": [233, 30, 99],
        "rectum": [121, 85, 72],
        "colon": [0, 121, 107],
        "femur_left": [255, 224, 178],
        "femur_right": [255, 224, 178],
        "hip_left": [230, 238, 156],
        "hip_right": [230, 238, 156],
        "sacrum": [141, 110, 99],
        "spinal_cord": [0, 255, 0],
        "thyroid_gland": [255, 105, 180],
        "skull": [255, 228, 196],
        "brain": [135, 206, 250],
        "common_carotid_artery_left": [220, 20, 60],
        "common_carotid_artery_right": [220, 20, 60],
        "superior_vena_cava": [70, 130, 180],
        "portal_vein_and_splenic_vein": [0, 139, 139],
        "clavicula_left": [244, 164, 96],
        "clavicula_right": [244, 164, 96],
        "sternum": [222, 184, 135],
        "iliac_artery_left": [255, 99, 71],
        "iliac_artery_right": [255, 99, 71]
    },
    "ru_names": {
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
}


class ContourEngine:
    """
    Сервисный класс, реализующий всю цепочку вычислений и постобработки автооконтурирования.
    """

    def __init__(self, config_path: str = "presets.json") -> None:
        self.config_path = Path(config_path).resolve()
        self.presets: Dict[str, List[str]] = {}
        self.colors: Dict[str, List[int]] = {}
        self.ru_names: Dict[str, str] = {}
        self.load_presets_config()

    def load_presets_config(self) -> None:
        """
        Загружает пресеты, цвета и переводы из presets.json.
        Если файл отсутствует, создает его с дефолтными значениями.
        """
        try:
            if not self.config_path.exists():
                logger.info(f"Файл конфигурации не найден. Создание дефолтного {self.config_path}...")
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_PRESETS_DATA, f, ensure_ascii=False, indent=2)
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.presets = data.get("presets", DEFAULT_PRESETS_DATA["presets"])
            self.colors = data.get("colors", DEFAULT_PRESETS_DATA["colors"])
            self.ru_names = data.get("ru_names", DEFAULT_PRESETS_DATA["ru_names"])
            logger.info("Конфигурация пресетов успешно загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить presets.json: {e}. Используются внутренние данные по умолчанию.")
            self.presets = DEFAULT_PRESETS_DATA["presets"]
            self.colors = DEFAULT_PRESETS_DATA["colors"]
            self.ru_names = DEFAULT_PRESETS_DATA["ru_names"]

    def save_presets_config(self) -> None:
        """
        Сохраняет текущую конфигурацию пресетов и цветов обратно в presets.json.
        """
        try:
            data = {
                "presets": self.presets,
                "colors": self.colors,
                "ru_names": self.ru_names
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Конфигурация пресетов успешно сохранена.")
        except Exception as e:
            logger.error(f"Не удалось сохранить presets.json: {e}")

    @staticmethod
    def is_gpu_available() -> bool:
        """
        Проверяет доступность GPU с поддержкой CUDA через PyTorch.
        """
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def remove_small_blobs(mask_3d: np.ndarray) -> np.ndarray:
        """
        Постобработка: оставляет только крупнейший 3D-связный компонент маски (основной объем органа),
        удаляя изолированные мелкие шумы нейросети.
        """
        labeled_array, num_features = label(mask_3d)
        if num_features <= 1:
            return mask_3d
        
        sizes = np.bincount(labeled_array.ravel())
        if len(sizes) <= 1:
            return mask_3d
            
        # Индекс самого большого компонента (sizes[0] - фон)
        largest_label = np.argmax(sizes[1:]) + 1
        return labeled_array == largest_label

    @staticmethod
    def smooth_3d_mask(mask_3d: np.ndarray, sigma: float) -> np.ndarray:
        """
        Постобработка: сглаживает ступенчатость контуров 3D-маски с помощью Гауссова фильтра.
        """
        if sigma <= 0.0:
            return mask_3d
        smoothed = gaussian_filter(mask_3d.astype(float), sigma=sigma)
        return smoothed > 0.5

    def verify_dicom_directory(self, dicom_dir: Path) -> int:
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
        self,
        dicom_dir_path: str,
        output_dir_path: str,
        preset_name: str,
        precision_mode: str = "normal",  # "normal", "fast", "faster"
        selected_organs: Optional[List[str]] = None,
        merge_mode: bool = False,
        existing_rtstruct_path: Optional[str] = None,
        use_gpu: bool = False,
        remove_blobs: bool = False,
        smoothing_sigma: float = 0.0,
        step_callback: Optional[Callable[[str], None]] = None,
        is_cancelled_cb: Optional[Callable[[], bool]] = None,
        register_process_cb: Optional[Callable[[subprocess.Popen], None]] = None
    ) -> None:
        """
        Основной пайплайн выполнения автооконтурирования органов риска на КТ.
        """
        start_time = time.time()
        dicom_dir = Path(dicom_dir_path).resolve()
        output_dir = Path(output_dir_path).resolve()
        
        # Инициализация временных путей внутри выходного каталога
        temp_dir = output_dir / "temp_autocontour_workspace"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        nifti_ct_path = temp_dir / "temp_ct_volume.nii.gz"
        segmentation_dir = temp_dir / "temp_masks"
        
        try:
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")
                
            # Проверка DICOM-файлов
            self.verify_dicom_directory(dicom_dir)
            
            # Считывание PatientID из первого DICOM-файла для динамического именования
            patient_id = "Unknown"
            try:
                dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
                if not dicom_files:
                    dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
                if dicom_files:
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
            
            d2n_settings.disable_validate_slice_increment()
            d2n_settings.disable_validate_orthogonal()
            d2n_settings.disable_validate_orientation()
            
            step_start = time.time()
            logger.info(f"Сборка 3D-тома NIfTI из {dicom_dir}... Это может занять некоторое время.")
            
            dicom2nifti.dicom_series_to_nifti(str(dicom_dir), str(nifti_ct_path), reorient_nifti=False)
            
            if not nifti_ct_path.exists():
                raise RuntimeError("Не удалось создать временный NIfTI-файл КТ.")
                
            logger.info(f"Шаг 1 успешно завершен за {time.time() - step_start:.2f} сек.")
            logger.info(f"Временный NIfTI сохранен: {nifti_ct_path} ({nifti_ct_path.stat().st_size / (1024*1024):.2f} МБ)")

            # ----------------------------------------------------------------------
            # Шаг 2: ИИ-сегментация через TotalSegmentator
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 2 из 5: Сегментация органов нейросетью TotalSegmentator...")
            logger.info("--- Шаг 2 из 5: ИИ-сегментация с помощью TotalSegmentator ---")
            
            step_start = time.time()
            
            # Логируем выбранный режим точности и вычислительное устройство
            device = "gpu" if (use_gpu and self.is_gpu_available()) else "cpu"
            logger.info(f"Параметры запуска: Режим точности: {precision_mode.upper()}, Устройство: {device.upper()}")
            
            segmentation_dir.mkdir(parents=True, exist_ok=True)
            
            # Получаем выбранные органы
            if selected_organs is not None:
                target_organs = selected_organs
                logger.info(f"ИИ сегментирует только выбранные OAR: {target_organs}")
            else:
                target_organs = self.presets.get(preset_name)
                if target_organs:
                    logger.info(f"ИИ сегментирует только выбранные OAR из пресета '{preset_name}': {target_organs}")
                else:
                    logger.warning(f"Пресет '{preset_name}' не найден. Будут экспортированы все найденные OAR.")
                    target_organs = None
            
            # Динамически получаем список поддерживаемых органов в TotalSegmentator
            supported_organs = set()
            try:
                from totalsegmentator.map_to_binary import class_map
                # Пытаемся получить карту классов
                if "total" in class_map:
                    supported_organs = set(class_map["total"].values())
                elif "total_v1" in class_map:
                    supported_organs = set(class_map["total_v1"].values())
                else:
                    for subset in class_map.values():
                        supported_organs.update(subset.values())
                logger.info(f"Динамически загружено классов TotalSegmentator: {len(supported_organs)}")
            except Exception as e:
                logger.warning(f"Не удалось динамически получить список классов TotalSegmentator: {e}. Резервный набор.")
                supported_organs = set(self.colors.keys())

            # Карта виртуальных органов, которые мы можем собрать из долей/частей
            VIRTUAL_ORGANS_MAP = {
                "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
                "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
            }

            # Адаптируем список целевых органов под поддерживаемые классы TotalSegmentator
            totalseg_rois = []
            if target_organs:
                for organ in target_organs:
                    if organ in supported_organs:
                        totalseg_rois.append(organ)
                    elif organ in VIRTUAL_ORGANS_MAP:
                        parts = VIRTUAL_ORGANS_MAP[organ]
                        supported_parts = [p for p in parts if p in supported_organs]
                        if supported_parts:
                            totalseg_rois.extend(supported_parts)
                            logger.info(f"Орган '{organ}' будет собран из частей: {supported_parts}")
                        else:
                            logger.warning(f"Орган '{organ}' задекларирован как виртуальный, но части не поддерживаются ИИ.")
                    else:
                        logger.warning(f"Орган '{organ}' не поддерживается текущей версией TotalSegmentator и будет пропущен.")
                
                totalseg_rois = sorted(list(set(totalseg_rois)))
                logger.info(f"Адаптированный список ROI для TotalSegmentator: {totalseg_rois}")

            # Находим путь к исполняемому файлу TotalSegmentator в виртуальном окружении
            exe_dir = Path(sys.executable).parent
            totalseg_exe = exe_dir / "TotalSegmentator.exe"
            if not totalseg_exe.exists():
                totalseg_exe = exe_dir / "TotalSegmentator"
            if not totalseg_exe.exists():
                totalseg_exe = Path("TotalSegmentator")
                
            cmd = [
                str(totalseg_exe),
                "-i", str(nifti_ct_path),
                "-o", str(segmentation_dir),
                "--device", device
            ]
            
            # Настройка флагов точности
            if precision_mode == "fast":
                cmd.append("--fast")
            elif precision_mode == "faster":
                # Режим ультра-быстрого поиска тела/суб-режима
                cmd.extend(["--fast", "--task", "body"])
                # Если в режиме body, то мы ищем только тело, но если пользователь передал конкретные ROI,
                # TotalSegmentator проигнорирует их. Поэтому мы логируем предупреждение.
                logger.warning("Запущен сверхбыстрый режим '--task body'. Сегментируется только контур тела!")
            
            # Передаем адаптированные органы, если это не режим body (в body ищется только тело)
            if precision_mode != "faster":
                if target_organs:
                    if totalseg_rois:
                        cmd.append("--roi_subset")
                        cmd.extend(totalseg_rois)
                    else:
                        raise RuntimeError(
                            "Ни один из выбранных органов не поддерживается текущей версией TotalSegmentator.\n"
                            "Пожалуйста, выберите другие органы риска (например, мочевой пузырь или кости)."
                        )
            
            logger.info(f"Запуск внешнего процесса TotalSegmentator: {' '.join(cmd)}")
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
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
            
            if register_process_cb:
                register_process_cb(process)
                
            # Чтение вывода в реальном времени с поддержкой мгновенной отмены
            while True:
                if is_cancelled_cb and is_cancelled_cb():
                    if process.poll() is None:
                        logger.info("Отмена: Принудительное завершение процесса TotalSegmentator...")
                        process.kill()
                    raise RuntimeError("Операция отменена пользователем.")
                    
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
            # Постобработка: сборка виртуальных органов (легкие)
            # ----------------------------------------------------------------------
            logger.info("--- Постобработка: Сборка цельных легких из долей ИИ ---")
            try:
                POST_VIRTUAL_MAP = {
                    "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
                    "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
                }
                
                for virtual_organ, parts in POST_VIRTUAL_MAP.items():
                    part_files = [segmentation_dir / f"{part}.nii.gz" for part in parts]
                    existing_part_files = [f for f in part_files if f.exists()]
                    
                    if existing_part_files:
                        logger.info(f"Сборка цельного органа '{virtual_organ}' из долей: {[f.name for f in existing_part_files]}")
                        base_nii = nib.load(str(existing_part_files[0]))
                        base_data = base_nii.get_fdata() > 0.5
                        
                        for part_file in existing_part_files[1:]:
                            part_data = nib.load(str(part_file)).get_fdata() > 0.5
                            base_data = base_data | part_data
                            
                        merged_nii = nib.Nifti1Image(base_data.astype(np.uint8), base_nii.affine, base_nii.header)
                        merged_file_path = segmentation_dir / f"{virtual_organ}.nii.gz"
                        nib.save(merged_nii, str(merged_file_path))
                        logger.info(f"Цельный орган успешно собран и сохранен как: {merged_file_path.name}")
                        
                        for part_file in existing_part_files:
                            try:
                                part_file.unlink()
                            except Exception as e:
                                logger.debug(f"Не удалось удалить файл части {part_file.name}: {e}")
            except Exception as e:
                logger.error(f"Не удалось завершить сборку виртуальных органов: {e}")

            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")

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

            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")

            # ----------------------------------------------------------------------
            # Шаг 4: Сборка масок в DICOM RTSTRUCT с 3D Постобработкой (Blobs / Smoothing)
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 4 из 5: Сборка RTSTRUCT и привязка к геометрии DICOM...")
            logger.info("--- Шаг 4 из 5: Сборка RTSTRUCT и привязка к геометрии DICOM ---")
            
            step_start = time.time()
            
            mask_files = list(segmentation_dir.glob("*.nii.gz"))
            if not mask_files:
                raise RuntimeError("Не найдено масок органов после сегментации.")
                
            detected_organs = sorted([f.name.replace(".nii.gz", "") for f in mask_files])
            logger.info(f"Обнаружено сегментированных масок органов: {len(mask_files)}")
            logger.info(f"Список определенных ИИ органов на КТ: {detected_organs}")
            
            existing_rois = []
            rtstruct = None
            
            if merge_mode and existing_rtstruct_path:
                rt_path = Path(existing_rtstruct_path)
                if rt_path.exists():
                    try:
                        logger.info(f"Загрузка существующего RTSTRUCT для слияния: {rt_path}")
                        rtstruct = RTStructBuilder.create_from(
                            dicom_series_path=str(dicom_dir),
                            rt_struct_path=str(rt_path)
                        )
                        existing_rois = rtstruct.get_roi_names()
                        logger.info(f"Существующие структуры в файле: {existing_rois}")
                    except Exception as e:
                        logger.error(
                            f"Не удалось загрузить RTSTRUCT '{rt_path}' для слияния: {e}. "
                            "Пайплайн переключен в режим создания НОВОГО файла."
                        )
                else:
                    logger.error(
                        f"Файл RTSTRUCT для слияния не найден на диске: '{rt_path}'. "
                        "Пайплайн переключен в режим создания НОВОГО файла."
                    )

            if rtstruct is None:
                logger.info("Инициализация нового RTSTRUCT считыванием оригинальной геометрии DICOM серии...")
                rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_dir))
                existing_rtstruct_path = None
            
            added_count = 0
            is_brachy = "Brachytherapy" in preset_name or preset_name == "brachytherapy_oar"
            
            for mask_file in mask_files:
                organ_name = mask_file.name.replace(".nii.gz", "")
                
                # Фильтруем по списку целевых органов (если это не сверхбыстрый режим body, где ищется всё тело)
                if precision_mode != "faster" and target_organs and organ_name not in target_organs:
                    continue
                    
                logger.info(f"Обработка органа: {organ_name}...")
                
                nii_mask = nib.load(str(mask_file))
                mask_data = nii_mask.get_fdata() > 0.5
                
                # ------------------------------------------------------------------
                # ПОСТОБРАБОТКА МАСОК (Remove small blobs)
                # ------------------------------------------------------------------
                if remove_blobs:
                    before_pixels = np.sum(mask_data)
                    mask_data = self.remove_small_blobs(mask_data)
                    after_pixels = np.sum(mask_data)
                    removed_pixels = before_pixels - after_pixels
                    if removed_pixels > 0:
                        logger.info(f"[{organ_name}] Удалено мелких артефактов (blobs): {removed_pixels} пикселей")

                # ------------------------------------------------------------------
                # ПОСТОБРАБОТКА МАСОК (Gaussian smoothing)
                # ------------------------------------------------------------------
                if smoothing_sigma > 0.0:
                    logger.info(f"[{organ_name}] Применение 3D-сглаживания Гаусса (sigma={smoothing_sigma})...")
                    mask_data = self.smooth_3d_mask(mask_data, smoothing_sigma)

                # Транспонируем (X, Y, Z) к NumPy (Y, X, Z) [Rows, Cols, Slices]
                mask_data_transposed = np.transpose(mask_data, (1, 0, 2))
                mask_bool = mask_data_transposed.astype(bool)
                
                if not np.any(mask_bool):
                    logger.info(f"Пропуск пустого органа: {organ_name} (отсутствует в КТ объеме после постобработки)")
                    continue
                    
                color = self.colors.get(organ_name, [128, 128, 128])
                
                # Локализация ROI строго на английском согласно ТЗ
                # Определяем имена ROI
                roi_names_to_add: List[Tuple[str, List[int]]] = []
                
                if is_brachy:
                    # Специальная кастомизация для пресета "Брахитерапия"
                    if organ_name == "urinary_bladder":
                        roi_names_to_add.append(("Bladder", color))
                    elif organ_name == "small_bowel":
                        roi_names_to_add.append(("Small Bowel", color))
                    elif organ_name == "colon":
                        # Дублируем маску кишечника под двумя именами (например, Colon и Colon Dup)
                        roi_names_to_add.append(("Colon", color))
                        # Цвет дубликата делаем немного отличающимся для визуального удобства
                        dup_color = [color[0], min(255, color[1] + 40), color[2]]
                        roi_names_to_add.append(("Colon Dup", dup_color))
                    else:
                        pretty_name = organ_name.replace("_", " ").title()
                        roi_names_to_add.append((pretty_name, color))
                else:
                    # Стандартное английское название
                    pretty_name = organ_name.replace("_", " ").title()
                    # Исключения для более компактного вида в TPS
                    if organ_name == "urinary_bladder":
                        pretty_name = "Bladder"
                    roi_names_to_add.append((pretty_name, color))
                
                # Добавление в RTSTRUCT
                for roi_name, roi_color in roi_names_to_add:
                    # Умное слияние с существующими контурами
                    final_name = roi_name
                    if final_name in existing_rois:
                        final_name = f"{final_name} (AI)"
                        logger.warning(f"Орган '{roi_name}' уже размечен. Добавлен как '{final_name}'")
                    
                    rtstruct.add_roi(
                        mask=mask_bool,
                        color=roi_color,
                        name=final_name
                    )
                    added_count += 1
                    logger.info(f"Успешно добавлен ROI '{final_name}' (цвет: {roi_color})")
                
            if added_count == 0:
                raise RuntimeError("В RTSTRUCT не было добавлено ни одного OAR. Проверьте область сканирования.")
                
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")

            # ----------------------------------------------------------------------
            # Шаг 5: Сохранение итогового файла
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 5 из 5: Запись итогового DICOM RTSTRUCT...")
            logger.info("--- Шаг 5 из 5: Запись итогового DICOM RTSTRUCT ---")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            clean_patient_id = "".join([c for c in str(patient_id) if c.isalnum() or c in ("_", "-")]).strip()
            if not clean_patient_id:
                clean_patient_id = "Unknown"

            if merge_mode and existing_rtstruct_path:
                orig_name = Path(existing_rtstruct_path).parent.name if Path(existing_rtstruct_path).stem == "rtstruct" else Path(existing_rtstruct_path).stem
                if orig_name.lower() == "rtstruct":
                    orig_name = Path(existing_rtstruct_path).parent.name
                rtstruct_filename = f"RTSTRUCT_{orig_name}_merged.dcm"
            else:
                rtstruct_filename = f"RTSTRUCT_{clean_patient_id}.dcm"

            rtstruct_file_path = output_dir / rtstruct_filename
            
            rtstruct.save(str(rtstruct_file_path))
            logger.info(f"Шаг 5 успешно завершен за {time.time() - step_start:.2f} сек.")
            logger.info(f"Итоговый файл RTSTRUCT успешно записан: {rtstruct_file_path}")
            
        except Exception as e:
            logger.error(f"Сбой в пайплайне: {e}", exc_info=True)
            logger.warning(f"Временная рабочая папка сохранена: {temp_dir}")
            raise e
            
        else:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
        finally:
            logger.info(f"Пайплайн завершен. Общее время работы: {time.time() - start_time:.2f} сек.")
