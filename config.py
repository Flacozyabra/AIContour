#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Модуль config.py: Глобальные конфигурационные константы и маппинги проекта AI Contour
================================================================================
"""

# Глобальный словарь маппинга органов на таски TotalSegmentator
ROI_TO_TASK_MAP = {
    # Отделы головного мозга (brain_structures)
    'brain': 'total',
    'brain_stem': 'brain_structures',
    'brainstem': 'brain_structures',
    'cerebellum': 'brain_structures',
    'thalamus_left': 'brain_structures',
    'thalamus_right': 'brain_structures',
    'hippocampus_left': 'brain_structures',
    'hippocampus_right': 'brain_structures',
    'amygdala_left': 'brain_structures',
    'amygdala_right': 'brain_structures',
    'caudate_left': 'brain_structures',
    'caudate_right': 'brain_structures',
    'putamen_left': 'brain_structures',
    'putamen_right': 'brain_structures',
    'pallidum_left': 'brain_structures',
    'pallidum_right': 'brain_structures',
    'subarachnoid_space': 'brain_structures',
    'venous_sinuses': 'brain_structures',
    'septum_pellucidum': 'brain_structures',
    'insular_cortex': 'brain_structures',
    'internal_capsule': 'brain_structures',
    'ventricle': 'brain_structures',
    'central_sulcus': 'brain_structures',
    'frontal_lobe': 'brain_structures',
    'parietal_lobe': 'brain_structures',
    'occipital_lobe': 'brain_structures',
    'temporal_lobe': 'brain_structures',

    # Мелкие органы головы (head_glands_cavities)
    'eye_left': 'head_glands_cavities',
    'eye_right': 'head_glands_cavities',
    'lens_left': 'head_glands_cavities',
    'lens_right': 'head_glands_cavities',
    'optic_nerve_left': 'head_glands_cavities',
    'optic_nerve_right': 'head_glands_cavities',
    'parotid_gland_left': 'head_glands_cavities',
    'parotid_gland_right': 'head_glands_cavities',
    'submandibular_gland_left': 'head_glands_cavities',
    'submandibular_gland_right': 'head_glands_cavities',
    'nasal_cavity_left': 'head_glands_cavities',
    'nasal_cavity_right': 'head_glands_cavities',
    'nasopharynx': 'head_glands_cavities',
    'oropharynx': 'head_glands_cavities',
    'hypopharynx': 'head_glands_cavities',
    'soft_palate': 'head_glands_cavities',
    'hard_palate': 'head_glands_cavities',
    'auditory_canal_left': 'head_glands_cavities',
    'auditory_canal_right': 'head_glands_cavities',
}

# Маппинг файлов масок ИИ на стандартные ID OAR
FILE_NAME_MAP = {
    "eye_lens_left": "lens_left",
    "eye_lens_right": "lens_right",
    "iliac_vena_left": "iliac_vein_left",
    "iliac_vena_right": "iliac_vein_right",
    "brainstem": "brain_stem"
}

# Жесткий маппинг красивых названий OAR для Elekta Monaco 5.51
MONACO_NAMES_MAP = {
    "lens_left": "Lens L",
    "lens_right": "Lens R",
    "optic_nerve_left": "Optic Nerve L",
    "optic_nerve_right": "Optic Nerve R",
    "urinary_bladder": "Bladder",
    "spinal_cord": "Spinal Cord",
    "lungs": "Lungs Total"
}

ORGAN_GROUPS = {
    "Внешний контур тела (Body)": [
        "body"
    ],
    "Голова и шея (Head & Neck)": [
        "brain", "eye_left", "eye_right", "lens_left", "lens_right", "optic_nerve_left", "optic_nerve_right",
        "spinal_cord", "thyroid_gland", "skull", "common_carotid_artery_left", "common_carotid_artery_right",
        "parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right",
        "nasal_cavity_left", "nasal_cavity_right", "nasopharynx", "oropharynx", "hypopharynx",
        "soft_palate", "hard_palate", "auditory_canal_left", "auditory_canal_right"
    ],
    "Грудная клетка (Thorax)": [
        "heart", "lung_left", "lung_right", "lungs", "trachea", "esophagus", "aorta", "pulmonary_artery",
        "superior_vena_cava", "sternum", "clavicula_left", "clavicula_right",
        "scapula_left", "scapula_right", "humerus_left", "humerus_right"
    ],
    "Брюшная полость (Abdomen)": [
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "duodenum",
        "adrenal_gland_left", "adrenal_gland_right", "portal_vein_and_splenic_vein", "small_bowel", "colon"
    ],
    "Малый таз (Pelvis)": [
        "urinary_bladder", "prostate", "sacrum", "hip_left", "hip_right", "femur_left", "femur_right",
        "iliac_artery_left", "iliac_artery_right", "iliac_vein_left", "iliac_vein_right",
        "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right"
    ],
    "Отделы головного мозга (Brain Structures)": [
        "brain_stem", "cerebellum", "thalamus_left", "thalamus_right", "caudate_left", "caudate_right",
        "putamen_left", "putamen_right", "pallidum_left", "pallidum_right",
        "ventricle", "subarachnoid_space", "venous_sinuses", "septum_pellucidum", "internal_capsule",
        "frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe", "insular_cortex"
    ]
}

# Словарь для нечеткого сопоставления (Fuzzy Matching) OAR из сторонних RTSTRUCT файлов
EXTERNAL_ALIASES = {
    # Внешний контур тела (Skin/Body/External)
    "body": "body",
    "external": "body",
    "skin": "body",
    
    # Глаза, хрусталики и орбиты
    "eyeleft": "eye_left",
    "eyel": "eye_left",
    "eyeright": "eye_right",
    "eyer": "eye_right",
    "eyes": "eye_left",
    "lensleft": "lens_left",
    "lensl": "lens_left",
    "lensright": "lens_right",
    "lensr": "lens_right",
    "lens": "lens_left",
    "eyelensleft": "lens_left",
    "eyelensl": "lens_left",
    "eyelensright": "lens_right",
    "eyelensr": "lens_right",
    
    # Зрительные нервы
    "opticnerveleft": "optic_nerve_left",
    "opticnervel": "optic_nerve_left",
    "opticnrvl": "optic_nerve_left",
    "opticnerveright": "optic_nerve_right",
    "opticnerver": "optic_nerve_right",
    "opticnrvr": "optic_nerve_right",
    
    # Спинной мозг и ствол мозга
    "spinalcord": "spinal_cord",
    "spinal": "spinal_cord",
    "brainstemprv": "brain_stem",
    
    # Слюнные, слезные и щитовидная железы
    "thyroidgland": "thyroid_gland",
    "thyroid": "thyroid_gland",
    "glndthyroid": "thyroid_gland",
    "parotidglandleft": "parotid_gland_left",
    "parotidleft": "parotid_gland_left",
    "parotidl": "parotid_gland_left",
    "parotidglandright": "parotid_gland_right",
    "parotidright": "parotid_gland_right",
    "parotidr": "parotid_gland_right",
    "submandibularglandleft": "submandibular_gland_left",
    "submandibularleft": "submandibular_gland_left",
    "submandibularl": "submandibular_gland_left",
    "glndsubmandl": "submandibular_gland_left",
    "glndsubmand": "submandibular_gland_left",
    "submandibularglandright": "submandibular_gland_right",
    "submandibularright": "submandibular_gland_right",
    "submandibularr": "submandibular_gland_right",
    "glndsubmandr": "submandibular_gland_right",
    "glndsublingl": "sublingual_gland_left",
    "glndsublingr": "sublingual_gland_right",
    "glndsubling": "sublingual_gland_left",
    
    # ЛОР-органы
    "larynx": "larynx",
    "nasalcavityleft": "nasal_cavity_left",
    "nasalcavityl": "nasal_cavity_left",
    "nasalcavityright": "nasal_cavity_right",
    "nasalcavityr": "nasal_cavity_right",
    "nasopharynx": "nasopharynx",
    "oropharynx": "oropharynx",
    "hypopharynx": "hypopharynx",
    "softpalate": "soft_palate",
    "hardpalate": "hard_palate",
    "auditorycanalleft": "auditory_canal_left",
    "auditorycanall": "auditory_canal_left",
    "auditorycanalright": "auditory_canal_right",
    "auditorycanalr": "auditory_canal_right",
    
    # Сосуды шеи и череп
    "skull": "skull",
    "commoncarotidarteryleft": "common_carotid_artery_left",
    "commoncarotidarteryl": "common_carotid_artery_left",
    "commoncarotidarteryright": "common_carotid_artery_right",
    "commoncarotidarteryr": "common_carotid_artery_right",
    
    # Грудная клетка
    "heart": "heart",
    "lungleft": "lung_left",
    "lungl": "lung_left",
    "lungright": "lung_right",
    "lungr": "lung_right",
    "lungs": "lung_left",
    "trachea": "trachea",
    "esophagus": "esophagus",
    "aorta": "aorta",
    "pulmonaryartery": "pulmonary_artery",
    "superiorvenacava": "superior_vena_cava",
    "sternum": "sternum",
    "claviculaleft": "clavicula_left",
    "claviculal": "clavicula_left",
    "claviclel": "clavicula_left",
    "clavicularight": "clavicula_right",
    "clavicular": "clavicula_right",
    "clavicler": "clavicula_right",
    "scapulaleft": "scapula_left",
    "scapulal": "scapula_left",
    "scapularight": "scapula_right",
    "scapular": "scapula_right",
    "humerusleft": "humerus_left",
    "humerusl": "humerus_left",
    "humerusright": "humerus_right",
    "humerusr": "humerus_right",
    
    # Брюшная полость
    "spleen": "spleen",
    "kidneyleft": "kidney_left",
    "kidneyl": "kidney_left",
    "kidneyright": "kidney_right",
    "kidneyr": "kidney_right",
    "kidneys": "kidney_left",
    "gallbladder": "gallbladder",
    "liver": "liver",
    "stomach": "stomach",
    "pancreas": "pancreas",
    "duodenum": "duodenum",
    "adrenalglandleft": "adrenal_gland_left",
    "adrenalglandl": "adrenal_gland_left",
    "adrenalglandright": "adrenal_gland_right",
    "adrenalglandr": "adrenal_gland_right",
    "portalveinandsplenicvein": "portal_vein_and_splenic_vein",
    "smallbowel": "small_bowel",
    "bowel": "small_bowel",
    "bowelbag": "small_bowel",
    "colon": "colon",
    
    # Малый таз
    "urinarybladder": "urinary_bladder",
    "bladder": "urinary_bladder",
    "prostate": "prostate",
    "rectum": "rectum",
    "anorectum": "rectum",
    "sacrum": "sacrum",
    "hipleft": "hip_left",
    "hipl": "hip_left",
    "boneilium": "hip_left",
    "bonepelvic": "hip_left",
    "hipright": "hip_right",
    "hipr": "hip_right",
    "femurleft": "femur_left",
    "femurl": "femur_left",
    "femurright": "femur_right",
    "femurr": "femur_right",
    "femurs": "femur_left",
    "iliacarteryleft": "iliac_artery_left",
    "iliacarteryl": "iliac_artery_left",
    "iliacarteryright": "iliac_artery_right",
    "iliacarteryr": "iliac_artery_right",
    "iliacveinleft": "iliac_vein_left",
    "iliacveinl": "iliac_vein_left",
    "iliacveinright": "iliac_vein_right",
    "iliacveinr": "iliac_vein_right",
    "gluteusmaximusleft": "gluteus_maximus_left",
    "gluteusmaximusl": "gluteus_maximus_left",
    "gluteusmaximusright": "gluteus_maximus_right",
    "gluteusmaximusr": "gluteus_maximus_right",
    "gluteusmediusleft": "gluteus_medius_left",
    "gluteusmediusl": "gluteus_medius_left",
    "gluteusmediusright": "gluteus_medius_right",
    "gluteusmediusr": "gluteus_medius_right",
    "gluteusminimusleft": "gluteus_minimus_left",
    "gluteusminimusl": "gluteus_minimus_left",
    "gluteusminimusright": "gluteus_minimus_right",
    "gluteusminimusr": "gluteus_minimus_right",
    
    # Отделы головного мозга
    "brain": "brain",
    "brainstem": "brain_stem",
    "brainstemtechnical": "brain_stem",
    "cerebellum": "cerebellum",
    "thalamusleft": "thalamus_left",
    "thalamusl": "thalamus_left",
    "thalamusright": "thalamus_right",
    "thalamusr": "thalamus_right",
    "hippocampusleft": "hippocampus_left",
    "hippocampusl": "hippocampus_left",
    "hippocampusright": "hippocampus_right",
    "hippocampusr": "hippocampus_right",
    "amygdalaleft": "amygdala_left",
    "amygdalal": "amygdala_left",
    "amygdalaright": "amygdala_right",
    "amygdalar": "amygdala_right",
    "caudateleft": "caudate_left",
    "caudatel": "caudate_left",
    "caudateright": "caudate_right",
    "caudater": "caudate_right",
    "putamenleft": "putamen_left",
    "putamenl": "putamen_left",
    "putamenright": "putamen_right",
    "putamenr": "putamen_right",
    "pallidumleft": "pallidum_left",
    "palliduml": "pallidum_left",
    "pallidumright": "pallidum_right",
    "pallidumr": "pallidum_right",
}

# Список суб-моделей ИИ, требующих отдельной академической или коммерческой лицензии
LICENSED_TASKS = {
    "brain_structures": "Отделы головного мозга (Brain Structures)"
}

import json
import threading
from pathlib import Path
from typing import Dict, Any, List

class StatisticsManager:
    """Класс для потокобезопасного управления статистикой автооконтурирования OAR."""
    
    def __init__(self, file_path: str = "config/statistics.json") -> None:
        import sys
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).parent
        else:
            base_dir = Path(__file__).parent.resolve()
        self.file_path = (base_dir / file_path).resolve()
        self._lock = threading.Lock()
        self._default_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "cancelled_runs": 0,
            "total_organs_contoured": 0,
            "total_elapsed_time_seconds": 0.0,
            "organ_stats": {},
            "recent_runs": []
        }
        
    def _load_stats(self) -> Dict[str, Any]:
        """Загружает статистику из JSON файла. Возвращает дефолтную структуру при отсутствии файла или ошибке."""
        if not self.file_path.exists():
            return self._default_stats.copy()
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Гарантируем наличие всех полей
                for key, val in self._default_stats.items():
                    if key not in data:
                        data[key] = val.copy() if isinstance(val, (dict, list)) else val
                return data
        except Exception:
            return self._default_stats.copy()

    def _save_stats(self, data: Dict[str, Any]) -> None:
        """Сохраняет статистику в JSON файл."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает текущую статистику (потокобезопасно)."""
        with self._lock:
            return self._load_stats()

    def reset_stats(self) -> None:
        """Сбрасывает всю статистику (потокобезопасно)."""
        with self._lock:
            self._save_stats(self._default_stats.copy())

    def record_run(
        self,
        status: str,
        elapsed_seconds: float,
        organs_contoured: List[str],
        preset_name: str,
        precision_mode: str
    ) -> None:
        """Записывает новый запуск автооконтурирования (потокобезопасно)."""
        with self._lock:
            data = self._load_stats()
            
            # Обновление общих счетчиков
            data["total_runs"] += 1
            if status == "success":
                data["successful_runs"] += 1
                data["total_organs_contoured"] += len(organs_contoured)
                data["total_elapsed_time_seconds"] += elapsed_seconds
                
                # Обновление статистики по каждому органу
                if "organ_stats" not in data or not isinstance(data["organ_stats"], dict):
                    data["organ_stats"] = {}
                for organ in organs_contoured:
                    data["organ_stats"][organ] = data["organ_stats"].get(organ, 0) + 1
            elif status == "cancelled":
                data["cancelled_runs"] += 1
            else:
                data["failed_runs"] += 1

            # Добавление в список последних запусков
            import datetime
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_entry = {
                "timestamp": now_str,
                "status": status,
                "preset": preset_name,
                "precision": precision_mode,
                "elapsed_seconds": round(elapsed_seconds, 1),
                "organs_count": len(organs_contoured) if status == "success" else 0
            }
            if "recent_runs" not in data or not isinstance(data["recent_runs"], list):
                data["recent_runs"] = []
            
            data["recent_runs"].insert(0, run_entry)
            # Ограничиваем список последних 20 запусками
            data["recent_runs"] = data["recent_runs"][:20]
            
            self._save_stats(data)

