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
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti
import dicom2nifti.settings as d2n_settings
import warnings
import rt_utils.rtstruct_builder
rt_utils.rtstruct_builder.warnings = warnings
from rt_utils import RTStructBuilder
from scipy.ndimage import label, gaussian_filter

# Настройка локального логера движка
logger = logging.getLogger("ContourEngine")

from config import ROI_TO_TASK_MAP, FILE_NAME_MAP, MONACO_NAMES_MAP, LICENSED_TASKS

# Дефолтные настройки для автогенерации presets.json при его отсутствии
DEFAULT_PRESETS_DATA = {
    "presets": {
        "Голова и шея (Head & Neck)": [
            "brain", "eye_left", "eye_right", "lens_left", "lens_right", "optic_nerve_left", "optic_nerve_right",
            "spinal_cord", "thyroid_gland", "skull", "common_carotid_artery_left", "common_carotid_artery_right",
            "parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right",
            "nasal_cavity_left", "nasal_cavity_right", "nasopharynx", "oropharynx", "hypopharynx",
            "soft_palate", "hard_palate", "auditory_canal_left", "auditory_canal_right"
        ],
        "Грудная клетка (Thorax)": [
            "heart", "lung_left", "lung_right", "trachea", "esophagus", "aorta", "pulmonary_artery",
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
            "brain_stem", "cerebellum", "thalamus_left", "thalamus_right",
            "caudate_left", "caudate_right", "putamen_left", "putamen_right",
            "pallidum_left", "pallidum_right", "ventricle", "subarachnoid_space", "venous_sinuses",
            "septum_pellucidum", "internal_capsule", "frontal_lobe", "parietal_lobe", "occipital_lobe",
            "temporal_lobe", "insular_cortex"
        ],
        "Остальное": []
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
        "lungs": [0, 172, 193],
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
        "iliac_artery_right": [255, 99, 71],
        "eye_left": [255, 255, 0],
        "eye_right": [255, 255, 0],
        "lens_left": [255, 165, 0],
        "lens_right": [255, 165, 0],
        "brain_stem": [210, 105, 30],
        "optic_nerve_left": [240, 230, 140],
        "optic_nerve_right": [240, 230, 140]
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
        "iliac_artery_left": "Левая подвздошная артерия (Iliac Artery L)",
        "iliac_artery_right": "Правая подвздошная артерия (Iliac Artery R)",
        "eye_left": "Левый глаз (Eye L)",
        "eye_right": "Правый глаз (Eye R)",
        "lens_left": "Левый хрусталик (Lens L)",
        "lens_right": "Правый хрусталик (Lens R)",
        "brain_stem": "Ствол мозга (Brain Stem)",
        "optic_nerve_left": "Левый зрительный нерв (Optic Nerve L)",
        "optic_nerve_right": "Правый зрительный нерв (Optic Nerve R)"
    }
}


class ContourEngine:
    """
    Сервисный класс, реализующий всю цепочку вычислений и постобработки автооконтурирования.
    """

    def __init__(self, config_path: str = "presets.json") -> None:
        self.config_path = Path(config_path).resolve()
        self.presets: Dict[str, List[str]] = {}
        self.preset_colors: Dict[str, Dict[str, List[int]]] = {}
        self.colors: Dict[str, List[int]] = {}
        self.ru_names: Dict[str, str] = {}
        self.load_presets_config()

    def _get_default_color(self, organ_name: str) -> List[int]:
        """Генерирует стабильный RGB цвет на основе хэша имени органа."""
        import hashlib
        h = hashlib.md5(organ_name.encode('utf-8')).digest()
        # Избегаем слишком темных цветов, минимальная яркость 50
        return [max(50, int(h[0])), max(50, int(h[1])), max(50, int(h[2]))]

    @staticmethod
    def translate_organ_to_ru(organ_name: str) -> str:
        """
        Интеллектуальный перевод названий органов TotalSegmentator на русский язык
        с сохранением английского аналога в скобках.
        """
        import re
        org = organ_name.lower().strip()
        
        # 1. Жестко заданные переводы для полных совпадений
        exact_translations = {
            "lungs": "Легкие (Lungs Total)",
            "brachiocephalic_trunk": "Плечеголовной ствол (Brachiocephalic Trunk)",
            "costal_cartilages": "Реберные хрящи (Costal Cartilages)",
            "middle_pharyngeal_constrictor": "Средний констриктор глотки (Middle Pharyngeal Constrictor)",
            "superior_pharyngeal_constrictor": "Верхний констриктор глотки (Superior Pharyngeal Constrictor)",
            "inferior_pharyngeal_constrictor": "Нижний констриктор глотки (Inferior Pharyngeal Constrictor)",
            "spleen": "Селезенка (Spleen)",
            "gallbladder": "Желчный пузырь (Gallbladder)",
            "liver": "Печень (Liver)",
            "stomach": "Желудок (Stomach)",
            "aorta": "Аорта (Aorta)",
            "inferior_vena_cava": "Нижняя полая вена (Vena Cava)",
            "superior_vena_cava": "Верхняя полая вена (Vena Cava Sup)",
            "portal_vein_and_splenic_vein": "Воротная и селезеночная вены (Portal/Splenic V)",
            "pancreas": "Поджелудочная железа (Pancreas)",
            "urinary_bladder": "Мочевой пузырь (Bladder)",
            "esophagus": "Пищевод (Esophagus)",
            "trachea": "Трахея (Trachea)",
            "heart": "Сердце (Heart)",
            "heart_myocardium": "Миокард сердца (Myocardium)",
            "pulmonary_artery": "Легочная артерия (Pulmonary Artery)",
            "brain": "Головной мозг (Brain)",
            "spinal_cord": "Спинной мозг (Spinal Cord)",
            "duodenum": "Двенадцатиперстная кишка (Duodenum)",
            "colon": "Ободочная кишка (Colon)",
            "rectum": "Прямая кишка (Rectum)",
            "small_bowel": "Тонкая кишка (Small Bowel)",
            "prostate": "Предстательная железа (Prostate)",
            "sacrum": "Крестец (Sacrum)",
            "sternum": "Грудина (Sternum)",
            "skull": "Череп (Skull)",
            "body": "Тело (Body)",
            "face": "Лицо (Face)",
            "thyroid_gland": "Щитовидная железа (Thyroid Gland)",
            "spinal_canal": "Спинномозговой канал (Spinal Canal)",
            "brain_stem": "Ствол мозга (Brain Stem)",
            "cerebellum": "Мозжечок (Cerebellum)",
            "ventricle_system": "Система желудочков мозга (Ventricles)",
            "ventricle": "Желудочки мозга (Ventricles)",
            "subarachnoid_space": "Субарахноидальное пространство (Subarachnoid Space)",
            "venous_sinuses": "Венозные синусы (Venous Sinuses)",
            "septum_pellucidum": "Прозрачная перегородка (Septum Pellucidum)",
            "insular_cortex": "Островковая кора (Insular Cortex)",
            "internal_capsule": "Внутренняя капсула (Internal Capsule)",
            "central_sulcus": "Центральная борозда (Central Sulcus)",
            "frontal_lobe": "Лобная доля (Frontal Lobe)",
            "parietal_lobe": "Теменная доля (Parietal Lobe)",
            "occipital_lobe": "Затылочная доля (Occipital Lobe)",
            "temporal_lobe": "Височная доля (Temporal Lobe)",
            "lingual_tonsil": "Язычная миндалина (Lingual Tonsil)",
            "nasopharynx": "Носоглотка (Nasopharynx)",
            "oropharynx": "Ротоглотка (Oropharynx)",
            "hypopharynx": "Гортаноглотка (Hypopharynx)",
            "larynx": "Гортань (Larynx)",
            "epiglottis": "Надгортанник (Epiglottis)",
            "hyoid_bone": "Подъязычная кость (Hyoid Bone)",
            "thyroid_cartilage": "Щитовидный хрящ (Thyroid Cartilage)",
            "cricoid_cartilage": "Перстневидный хрящ (Cricoid Cartilage)",
            "teeth": "Зубы (Teeth)",
            "soft_palate": "Мягкое нёбо (Soft Palate)",
            "hard_palate": "Твёрдое нёбо (Hard Palate)",
        }
        
        if org in exact_translations:
            return exact_translations[org]
            
        # 2. Регулярное выражение для ребер (rib_left_1..12, rib_right_1..12)
        rib_match = re.match(r"^rib_(left|right)_(\d+)$", org)
        if rib_match:
            side = rib_match.group(1)
            num = rib_match.group(2)
            side_ru = "Левое" if side == "left" else "Правое"
            side_en = "L" if side == "left" else "R"
            return f"{side_ru} ребро {num} (Rib {side_en}{num})"
            
        # 3. Регулярное выражение для позвонков (vertebrae_C1..C7, vertebrae_T1..T12, vertebrae_L1..L5, vertebrae_S1..S5)
        vert_match = re.match(r"^vertebrae_([cctlsa-zA-Z])(\d+)$", org)
        if vert_match:
            v_type = vert_match.group(1).upper()
            num = vert_match.group(2)
            type_ru = "Шейный"
            if v_type == "T":
                type_ru = "Грудной"
            elif v_type == "L":
                type_ru = "Поясничный"
            elif v_type == "S":
                type_ru = "Крестцовый"
            return f"{type_ru} позвонок {v_type}{num} (Vertebra {v_type}{num})"
            
        # 4. Регулярное выражение для парных костей, сосудов, мышц и органов с суффиксами _left и _right
        side_match = re.match(r"^(.+)_(left|right)$", org)
        if side_match:
            base = side_match.group(1)
            side = side_match.group(2)
            
            # Словарь базовых анатомических структур и их родов (m = мужской, f = женский, n = средний)
            base_definitions = {
                "nasal_cavity": ("носовая полость", "Nasal Cavity", "f"),
                "auditory_canal": ("слуховой проход", "Auditory Canal", "m"),
                "medial_pterygoid": ("медиальная крыловидная мышца", "Medial Pterygoid", "f"),
                "lateral_pterygoid": ("латеральная крыловидная мышца", "Lateral Pterygoid", "f"),
                "medial_rectus_muscle": ("медиальная прямая мышца глаза", "Medial Rectus Muscle", "f"),
                "lateral_rectus_muscle": ("латеральная прямая мышца глаза", "Lateral Rectus Muscle", "f"),
                "superior_rectus_muscle": ("верхняя прямая мышца глаза", "Superior Rectus Muscle", "f"),
                "inferior_rectus_muscle": ("нижняя прямая мышца глаза", "Inferior Rectus Muscle", "f"),
                "superior_oblique_muscle": ("верхняя косая мышца глаза", "Superior Oblique Muscle", "f"),
                "inferior_oblique_muscle": ("нижняя косая мышца глаза", "Inferior Oblique Muscle", "f"),
                "levator_palpebrae_superioris": ("мышца, поднимающая верхнее веко", "Levator Palpebrae", "f"),
                "middle_scalene": ("средняя лестничная мышца", "Middle Scalene", "f"),
                "anterior_scalene": ("передняя лестничная мышца", "Anterior Scalene", "f"),
                "posterior_scalene": ("задняя лестничная мышца", "Posterior Scalene", "f"),
                "prevertebral": ("предпозвоночная мышца", "Prevertebral Muscle", "f"),
                "pectoralis_major": ("большая грудная мышца", "Pectoralis Major", "f"),
                "pectoralis_minor": ("малая грудная мышца", "Pectoralis Minor", "f"),
                "serratus_anterior": ("передняя зубчатая мышца", "Serratus Anterior", "f"),
                "subscapularis": ("подлопаточная мышца", "Subscapularis", "f"),
                "supraspinatus": ("надостная мышца", "Supraspinatus", "f"),
                "infraspinatus": ("подостная мышца", "Infraspinatus", "f"),
                "teres_major": ("большая круглая мышца", "Teres Major", "f"),
                "teres_minor": ("малая круглая мышца", "Teres Minor", "f"),
                "latissimus_dorsi": ("широчайшая мышца спины", "Latissimus Dorsi", "f"),
                "trapezius": ("трапециевидная мышца", "Trapezius", "f"),
                "deltoid": ("дельтовидная мышца", "Deltoid", "f"),
                "quadratus_lumborum": ("квадратная мышца поясницы", "Quadratus Lumborum", "f"),
                "sartorius": ("портняжная мышца", "Sartorius", "f"),
                "brachiocephalic_vein": ("плечеголовная вена", "Brachiocephalic Vein", "f"),
                "iliac_artery": ("подвздошная артерия", "Iliac Artery", "f"),
                "iliac_vein": ("подвздошная вена", "Iliac Vein", "f"),
                # Органы
                "kidney": ("почка", "Kidney", "f"),
                "lung": ("легкое", "Lung", "n"),
                "adrenal_gland": ("надпочечник", "Adrenal Gland", "m"),
                "eye": ("глаз", "Eye", "m"),
                "lens": ("хрусталик", "Lens", "m"),
                "optic_nerve": ("зрительный нерв", "Optic Nerve", "m"),
                
                # Слюнные и другие железы
                "submandibular_gland": ("поднижнечелюстная слюнная железа", "Submandibular Gland", "f"),
                "parotid_gland": ("околоушная слюнная железа", "Parotid Gland", "f"),
                "sublingual_gland": ("подъязычная слюнная железа", "Sublingual Gland", "f"),
                "parathyroid_gland": ("околощитовидная железа", "Parathyroid Gland", "f"),

                # Мозг и нервная система
                "caudate_nucleus": ("хвостатое ядро", "Caudate Nucleus", "n"),
                "caudate": ("хвостатое ядро", "Caudate", "n"),
                "putamen": ("скорлупа мозга", "Putamen", "f"),
                "thalamus": ("таламус", "Thalamus", "m"),
                "globus_pallidus": ("бледный шар", "Globus Pallidus", "m"),
                "pallidum": ("бледный шар", "Pallidum", "m"),
                "amygdala": ("миндалевидное тело", "Amygdala", "n"),
                "hippocampus": ("гиппокамп", "Hippocampus", "m"),
                "internal_capsule": ("внутренняя капсула", "Internal Capsule", "f"),
                "ventricle": ("желудочек мозга", "Ventricle", "m"),

                # Кости черепа и лица
                "nasal_bone": ("носовая кость", "Nasal Bone", "f"),
                "lacrimal_bone": ("слезная кость", "Lacrimal Bone", "f"),
                "palatine_bone": ("небная кость", "Palatine Bone", "f"),
                "zygomatic_arch": ("скуловая дуга", "Zygomatic Arch", "f"),
                "mandible": ("нижняя челюсть", "Mandible", "f"),
                "maxilla": ("верхняя челюсть", "Maxilla", "f"),

                # Мышцы шеи и головы
                "masseter": ("жевательная мышца", "Masseter", "f"),
                "temporalis": ("височная мышца", "Temporalis", "f"),
                "buccinator": ("щечная мышца", "Buccinator", "f"),
                "pterygoid_medial": ("медиальная крыловидная мышца", "Medial Pterygoid", "f"),
                "pterygoid_lateral": ("латеральная крыловидная мышца", "Lateral Pterygoid", "f"),
                "digastric": ("двубрюшная мышца", "Digastric", "f"),
                "mylohyoid": ("челюстно-подъязычная мышца", "Mylohyoid", "f"),
                "geniohyoid": ("подбородочно-подъязычная мышца", "Geniohyoid", "f"),
                "sternohyoid": ("грудино-подъязычная мышца", "Sternohyoid", "f"),
                "omohyoid": ("лопаточно-подъязычная мышца", "Omohyoid", "f"),
                "thyrohyoid": ("щитоподъязычная мышца", "Thyrohyoid", "f"),
                "sternothyroid": ("грудино-щитовидная мышца", "Sternothyroid", "f"),
                "platysma": ("подкожная мышца шеи (платизма)", "Platysma", "f"),
                "sternocleidomastoid": ("грудино-ключично-сосцевидная мышца", "Sternocleidomastoid", "f"),

                # Сосуды головы и шеи
                "internal_carotid_artery": ("внутренняя сонная артерия", "Internal Carotid A", "f"),
                "external_carotid_artery": ("наружная сонная артерия", "External Carotid A", "f"),
                "vertebral_artery": ("позвоночная артерия", "Vertebral Artery", "f"),
                "external_jugular_vein": ("наружная яремная вена", "External Jugular Vein", "f"),
                
                # Кости
                "humerus": ("плечевая кость", "Humerus", "f"),
                "scapula": ("лопатка", "Scapula", "f"),
                "clavicula": ("ключица", "Clavicle", "f"),
                "femur": ("бедренная кость", "Femur", "f"),
                "hip": ("тазовая кость (таз)", "Hip", "f"),
                "patella": ("надколенник", "Patella", "m"),
                "tibia": ("большеберцовая кость", "Tibia", "f"),
                "fibula": ("малоберцовая кость", "Fibula", "f"),
                
                # Сосуды
                "subclavian_artery": ("подключичная артерия", "Subclavian Artery", "f"),
                "subclavian_vein": ("подключичная вена", "Subclavian Vein", "f"),
                "common_carotid_artery": ("общая сонная артерия", "Carotid Artery", "f"),
                "internal_jugular_vein": ("внутренняя яремная вена", "Jugular Vein", "f"),
                "iliac_artery": ("подвздошная артерия", "Iliac Artery", "f"),
                "iliac_vein": ("подвздошная вена", "Iliac Vein", "f"),
                
                # Мышцы
                "gluteus_maximus": ("большая ягодичная мышца", "Gluteus Maximus", "f"),
                "gluteus_medius": ("средняя ягодичная мышца", "Gluteus Medius", "f"),
                "gluteus_minimus": ("малая ягодичная мышца", "Gluteus Minimus", "f"),
                "autochthon": ("автохтонная мышца спины", "Autochthon Muscle", "f"),
                "iliopsoas": ("подвздошно-поясничная мышца", "Iliopsoas Muscle", "f"),
                "psoas_major": ("большая поясничная мышца", "Psoas Major Muscle", "f"),
                
                # Дополнительно
                "heart_atrium": ("предсердие", "Atrium", "n"),
                "heart_ventricle": ("желудочек", "Ventricle", "m"),
                "palatine_tonsil": ("небная миндалина", "Palatine Tonsil", "f"),
                "vocal_cord": ("голосовая связка", "Vocal Cord", "f"),
                "vestibular_fold": ("ложная голосовая связка", "Vestibular Fold", "f"),

                # Фаланги пальцев и другие кости конечностей
                "phalanx_proximal": ("проксимальная фаланга", "Phalanx Proximal", "f"),
                "phalanx_middle": ("средняя фаланга", "Phalanx Middle", "f"),
                "phalanx_distal": ("дистальная фаланга", "Phalanx Distal", "f"),
                "metacarpal": ("пястная кость", "Metacarpal", "f"),
                "metatarsal": ("плюсневая кость", "Metatarsal", "f"),
            }
            
            if base in base_definitions:
                ru_base, en_base, gender = base_definitions[base]
                
                # Согласование по роду
                if side == "left":
                    side_ru = "Левая" if gender == "f" else ("Левый" if gender == "m" else "Левое")
                    side_en = "L"
                else:
                    side_ru = "Правая" if gender == "f" else ("Правый" if gender == "m" else "Правое")
                    side_en = "R"
                    
                return f"{side_ru} {ru_base} ({en_base} {side_en})"
                
        # Если ничего не подошло, возвращаем красивую капитализированную строку с оригинальным именем
        clean_org = organ_name.replace("_", " ").title()
        return f"{clean_org} ({organ_name})"

    def _update_presets_with_total_classes(self) -> None:
        """Сравнивает текущие ru_names/colors с полным списком и дополняет их."""
        all_organs = self.get_all_supported_organs()
        if not all_organs:
            all_organs = []
        else:
            all_organs = list(all_organs)

        from config import ORGAN_GROUPS
        for group, organs in ORGAN_GROUPS.items():
            for org in organs:
                if org not in all_organs:
                    all_organs.append(org)

        changed = False
        full_total_preset = []
        for org in all_organs:
            if org == "body":
                continue
            full_total_preset.append(org)
            if org not in self.ru_names or self.ru_names[org] == org:
                self.ru_names[org] = self.translate_organ_to_ru(org)
                changed = True
            if org not in self.colors:
                self.colors[org] = self._get_default_color(org)
                changed = True

        # Проверим также все существующие ключи в self.ru_names на случай, если там остались латинские дубли
        for org in list(self.ru_names.keys()):
            # Проверяем, если значение совпадает с ключом ИЛИ не содержит ни одной кириллической буквы (остался латинский дубль)
            has_cyrillic = any(u'\u0400' <= char <= u'\u04FF' for char in self.ru_names[org])
            if self.ru_names[org] == org or not has_cyrillic:
                self.ru_names[org] = self.translate_organ_to_ru(org)
                changed = True

        # Гарантируем переводы и дефолтные цвета для всех структур, объявленных в ORGAN_GROUPS из config
        from config import ORGAN_GROUPS
        for group, organs in ORGAN_GROUPS.items():
            for org in organs:
                has_cyrillic_org = org in self.ru_names and any(u'\u0400' <= char <= u'\u04FF' for char in self.ru_names[org])
                if org not in self.ru_names or self.ru_names[org] == org or not has_cyrillic_org:
                    self.ru_names[org] = self.translate_organ_to_ru(org)
                    changed = True
                if org not in self.colors:
                    self.colors[org] = self._get_default_color(org)
                    changed = True

        # Обновляем пресеты до 6 строгих групп согласно ТЗ + Full Total
        head_neck_base = [
            "brain", "eye_left", "eye_right", "lens_left", "lens_right", "optic_nerve_left", "optic_nerve_right",
            "spinal_cord", "thyroid_gland", "skull", "common_carotid_artery_left", "common_carotid_artery_right",
            "parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right",
            "nasal_cavity_left", "nasal_cavity_right", "nasopharynx", "oropharynx", "hypopharynx",
            "soft_palate", "hard_palate", "auditory_canal_left", "auditory_canal_right"
        ]
        thorax_base = [
            "heart", "lung_left", "lung_right", "trachea", "esophagus", "aorta", "pulmonary_artery",
            "superior_vena_cava", "sternum", "clavicula_left", "clavicula_right",
            "scapula_left", "scapula_right", "humerus_left", "humerus_right"
        ]
        abdomen_base = [
            "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas", "duodenum",
            "adrenal_gland_left", "adrenal_gland_right", "portal_vein_and_splenic_vein", "small_bowel", "colon"
        ]
        pelvis_base = [
            "urinary_bladder", "prostate", "rectum", "sacrum", "hip_left", "hip_right", "femur_left", "femur_right",
            "iliac_artery_left", "iliac_artery_right", "iliac_vein_left", "iliac_vein_right",
            "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
            "gluteus_minimus_left", "gluteus_minimus_right"
        ]
        brain_structs_base = [
            "brain_stem", "cerebellum", "thalamus_left", "thalamus_right", "hippocampus_left", "hippocampus_right",
            "amygdala_left", "amygdala_right", "caudate_left", "caudate_right", "putamen_left", "putamen_right",
            "pallidum_left", "pallidum_right", "ventricle", "subarachnoid_space", "venous_sinuses",
            "septum_pellucidum", "internal_capsule", "frontal_lobe", "parietal_lobe", "occipital_lobe",
            "temporal_lobe", "insular_cortex", "central_sulcus"
        ]

        # Эвристические списки ключевых слов для автоматического распределения 300+ дополнительных структур
        brain_keywords = ['ventricle', 'thalamus', 'caudate', 'putamen', 'pallidum', 'cerebellum', 'cortex', 'capsule', 'hemorrhage', 'subarachnoid', 'temporal_lobe', 'occipital_lobe', 'frontal_lobe', 'parietal_lobe', 'lentiform']
        head_neck_keywords = ['eye', 'lens', 'optic', 'thyroid', 'skull', 'carotid', 'jugular', 'parotid', 'submandibular', 'nasal', 'nasopharynx', 'oropharynx', 'hypopharynx', 'palate', 'auditory', 'mandible', 'maxilla', 'jawbone', 'teeth', 'tooth', 'tongue', 'zygomatic', 'vocal_cords', 'larynx', 'pharynx', 'masseter', 'temporalis', 'buccinator', 'pterygoid', 'digastric', 'mylohyoid', 'geniohyoid', 'sternohyoid', 'omohyoid', 'thyrohyoid', 'sternothyroid', 'platysma', 'prevertebral', 'scalene', 'longus', 'capitis', 'sinus', 'alveolar_canal', 'incisive_canal', 'gland', 'face', 'head', 'rectus_muscle', 'oblique_muscle', 'levator_palpebrae', 'constrictor', 'cartilage', 'hyoid', 'vocal']
        thorax_keywords = ['heart', 'lung', 'trachea', 'esophagus', 'aorta', 'pulmonary', 'superior_vena_cava', 'sternum', 'clavicula', 'clavicle', 'scapula', 'humerus', 'pericardium', 'mediastinum', 'thoracic_cavity', 'subclavian', 'brachiocephalic', 'atrial_appendage', 'left_coronary_cusp', 'right_coronary_cusp', 'non_coronary_cusp', 'coronary', 'pleural', 'bronchia', 'thymus', 'breast']
        abdomen_keywords = ['spleen', 'kidney', 'gallbladder', 'liver', 'stomach', 'pancreas', 'duodenum', 'adrenal', 'portal_vein', 'splenic_vein', 'small_bowel', 'colon', 'abdominal_cavity', 'inferior_vena_cava', 'celiac', 'mesenteric', 'gastric', 'splenic_artery', 'hepatic', 'renal', 'biliary', 'pancreatic', 'duodenal']
        pelvis_keywords = ['bladder', 'prostate', 'rectum', 'sacrum', 'hip', 'femur', 'iliac', 'gluteus', 'obturator', 'piriformis', 'levator_ani', 'sphincter', 'urethra', 'seminal_vesicle', 'uterus', 'ovary', 'vagina', 'penis', 'testicle', 'pubic', 'coccyx', 'sartorius', 'pectineus', 'gracilis', 'adductor', 'quadriceps', 'hamstring', 'psoas', 'iliopsoas', 'quadratus_lumborum']

        head_neck = []
        thorax = []
        abdomen = []
        pelvis = []
        brain_structs = []
        other_preset = []

        for org in all_organs:
            if org == "body":
                continue
            
            # 1. Сначала жесткие базовые пресеты (для сохранения красивого Monaco порядка)
            if org in head_neck_base:
                head_neck.append(org)
            elif org in thorax_base:
                thorax.append(org)
            elif org in abdomen_base:
                abdomen.append(org)
            elif org in pelvis_base:
                pelvis.append(org)
            elif org in brain_structs_base:
                brain_structs.append(org)
                
            # 2. Иначе распределяем по эвристикам
            elif 'adrenal_gland' in org:
                abdomen.append(org)
            elif any(k in org for k in ['heart_ventricle', 'heart_atrium', 'heart_myocardium']):
                thorax.append(org)
            elif any(k in org for k in head_neck_keywords):
                head_neck.append(org)
            elif any(k in org for k in thorax_keywords):
                thorax.append(org)
            elif any(k in org for k in abdomen_keywords):
                abdomen.append(org)
            elif any(k in org for k in pelvis_keywords):
                pelvis.append(org)
            else:
                other_preset.append(org)

        new_presets = {
            "Голова и шея (Head & Neck)": head_neck,
            "Грудная клетка (Thorax)": thorax,
            "Брюшная полость (Abdomen)": abdomen,
            "Малый таз (Pelvis)": pelvis,
            "Отделы головного мозга (Brain Structures)": brain_structs,
        }

        # Дополняем отсутствующие пресеты, но не затираем существующие кастомизированные
        for k, v in new_presets.items():
            if k not in self.presets:
                self.presets[k] = v
                changed = True

        if changed:
            logger.info("Обнаружены изменения или новые структуры. Обновление конфигурации в папке config/...")
            self.save_presets_config()


    def _get_preset_filename(self, name: str) -> str:
        """Преобразует название пресета в безопасное имя файла на латинице."""
        mapping = {
            "Голова и шея (Head & Neck)": "head_and_neck",
            "Грудная клетка (Thorax)": "thorax",
            "Брюшная полость (Abdomen)": "abdomen",
            "Малый таз (Pelvis)": "pelvis",
            "Отделы головного мозга (Brain Structures)": "brain_structures"
        }
        if name in mapping:
            return mapping[name]
        
        # Для других/пользовательских пресетов
        clean = name.lower()
        translit_dict = {
            'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo',
            'ж':'zh','з':'z','и':'i','й':'y','к':'k','л':'l','м':'m',
            'н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u',
            'ф':'f','х':'kh','ц':'ts','ч':'ch','ш':'sh','щ':'sch',
            'ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'
        }
        for ru_c, en_c in translit_dict.items():
            clean = clean.replace(ru_c, en_c)
        
        clean = re.sub(r'[^a-z0-9_\- ]', '', clean)
        clean = clean.replace(' ', '_')
        clean = re.sub(r'_+', '_', clean).strip('_')
        return clean or "custom_preset"

    def _migrate_old_presets(self, old_presets_path: Path) -> None:
        """Переносит данные из монолитного presets.json в новую модульную структуру config/."""
        logger.info("Обнаружен старый presets.json в корне. Выполняется автоматическая миграция...")
        try:
            with open(old_presets_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.presets_dir.mkdir(parents=True, exist_ok=True)
            
            # Цвета
            colors = old_data.get("colors", DEFAULT_PRESETS_DATA["colors"])
            with open(self.colors_path, "w", encoding="utf-8") as f:
                json.dump(colors, f, ensure_ascii=False, indent=2)
                
            # Переводы
            ru_names = old_data.get("ru_names", DEFAULT_PRESETS_DATA["ru_names"])
            with open(self.translations_path, "w", encoding="utf-8") as f:
                json.dump(ru_names, f, ensure_ascii=False, indent=2)
                
            # Лицензии
            raw_licenses = old_data.get("licenses", "")
            if isinstance(raw_licenses, dict):
                licenses_val = next((v for v in raw_licenses.values() if v), "")
            else:
                licenses_val = str(raw_licenses)
            with open(self.licenses_path, "w", encoding="utf-8") as f:
                json.dump({"license_key": licenses_val}, f, ensure_ascii=False, indent=2)
                
            # Пресеты
            presets = old_data.get("presets", DEFAULT_PRESETS_DATA["presets"])
            for name, organs in presets.items():
                file_name = self._get_preset_filename(name)
                preset_filepath = self.presets_dir / f"{file_name}.json"
                with open(preset_filepath, "w", encoding="utf-8") as f:
                    json.dump({"name": name, "organs": organs}, f, ensure_ascii=False, indent=2)
                    
            # Резервное копирование и удаление
            bak_path = old_presets_path.with_suffix(".json.bak")
            if bak_path.exists():
                bak_path.unlink()
            old_presets_path.rename(bak_path)
            logger.info(f"Миграция успешно завершена. Резервная копия сохранена в {bak_path.name}")
        except Exception as e:
            logger.error(f"Не удалось завершить автоматическую миграцию: {e}")

    def load_presets_config(self) -> None:
        """
        Загружает пресеты, цвета, переводы и лицензии из папки config/.
        Если папка или файлы отсутствуют, выполняет миграцию или генерирует дефолтные файлы.
        """
        try:
            self.config_dir = Path("config").resolve()
            self.presets_dir = self.config_dir / "presets"
            self.colors_path = self.config_dir / "colors.json"
            self.translations_path = self.config_dir / "translations.json"
            self.licenses_path = self.config_dir / "licenses.json"
            
            # 1. Плавная миграция
            old_presets_path = Path("presets.json").resolve()
            if old_presets_path.exists() and not self.config_dir.exists():
                self._migrate_old_presets(old_presets_path)
                
            # 2. Создание директорий, если они не существуют
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.presets_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. Загрузка цветов
            if self.colors_path.exists():
                with open(self.colors_path, "r", encoding="utf-8") as f:
                    self.colors = json.load(f)
            else:
                logger.info(f"Файл цветов не найден. Создание {self.colors_path}...")
                self.colors = DEFAULT_PRESETS_DATA["colors"].copy()
                with open(self.colors_path, "w", encoding="utf-8") as f:
                    json.dump(self.colors, f, ensure_ascii=False, indent=2)
                
            # 4. Загрузка переводов (локализации)
            if self.translations_path.exists():
                with open(self.translations_path, "r", encoding="utf-8") as f:
                    self.ru_names = json.load(f)
            else:
                logger.info(f"Файл переводов не найден. Создание {self.translations_path}...")
                self.ru_names = DEFAULT_PRESETS_DATA["ru_names"].copy()
                with open(self.translations_path, "w", encoding="utf-8") as f:
                    json.dump(self.ru_names, f, ensure_ascii=False, indent=2)
                
            # 5. Загрузка лицензий
            if self.licenses_path.exists():
                with open(self.licenses_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.licenses = data.get("license_key", "")
            else:
                self.licenses = ""
                with open(self.licenses_path, "w", encoding="utf-8") as f:
                    json.dump({"license_key": ""}, f, ensure_ascii=False, indent=2)
                
            # 6. Загрузка пресетов
            self.presets = {}
            self.preset_colors = {}
            preset_files = list(self.presets_dir.glob("*.json"))
            if preset_files:
                for p_file in preset_files:
                    try:
                        with open(p_file, "r", encoding="utf-8") as f:
                            preset_data = json.load(f)
                            if isinstance(preset_data, dict) and "name" in preset_data and "organs" in preset_data:
                                name = preset_data["name"]
                                self.presets[name] = preset_data["organs"]
                                if "colors" in preset_data and isinstance(preset_data["colors"], dict):
                                    self.preset_colors[name] = preset_data["colors"]
                    except Exception as pe:
                        logger.error(f"Ошибка при загрузке пресета {p_file.name}: {pe}")
            
            # Если пресеты не были загружены, создаем дефолтные
            if not self.presets:
                logger.info("Пресеты не найдены. Генерация дефолтных пресетов...")
                self.presets = DEFAULT_PRESETS_DATA["presets"].copy()
                for name, organs in self.presets.items():
                    file_name = self._get_preset_filename(name)
                    p_file = self.presets_dir / f"{file_name}.json"
                    with open(p_file, "w", encoding="utf-8") as f:
                        json.dump({"name": name, "organs": organs}, f, ensure_ascii=False, indent=2)
            
            logger.info("Конфигурация пресетов успешно загружена.")
            
            # Динамическое дополнение до 117 классов TotalSegmentator
            self._update_presets_with_total_classes()
            
        except Exception as e:
            logger.error(f"Не удалось загрузить конфигурацию: {e}. Используются внутренние данные по умолчанию.")
            self.presets = DEFAULT_PRESETS_DATA["presets"].copy()
            self.colors = DEFAULT_PRESETS_DATA["colors"].copy()
            self.ru_names = DEFAULT_PRESETS_DATA["ru_names"].copy()
            self.licenses = ""

    def save_presets_config(self) -> None:
        """
        Сохраняет текущую конфигурацию пресетов, цветов, переводов и лицензий в config/.
        """
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.presets_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Сохранение цветов
            with open(self.colors_path, "w", encoding="utf-8") as f:
                json.dump(self.colors, f, ensure_ascii=False, indent=2)
                
            # 2. Сохранение локализации
            with open(self.translations_path, "w", encoding="utf-8") as f:
                json.dump(self.ru_names, f, ensure_ascii=False, indent=2)
                
            # 3. Сохранение лицензии
            with open(self.licenses_path, "w", encoding="utf-8") as f:
                json.dump({"license_key": getattr(self, "licenses", "")}, f, ensure_ascii=False, indent=2)
                
            # 4. Сохранение пресетов
            # Сначала найдем все текущие файлы в presets_dir, чтобы удалить файлы пресетов, которых больше нет
            current_files = {f"{self._get_preset_filename(name)}.json" for name in self.presets.keys()}
            for existing_file in self.presets_dir.glob("*.json"):
                if existing_file.name not in current_files:
                    try:
                        existing_file.unlink()
                    except Exception as e:
                        logger.warning(f"Не удалось удалить устаревший файл пресета {existing_file.name}: {e}")
            
            # Записываем пресеты в файлы
            for name, organs in self.presets.items():
                file_name = self._get_preset_filename(name)
                p_file = self.presets_dir / f"{file_name}.json"
                
                preset_payload = {
                    "name": name,
                    "organs": organs
                }
                if name in self.preset_colors:
                    preset_payload["colors"] = self.preset_colors[name]
                    
                with open(p_file, "w", encoding="utf-8") as f:
                    json.dump(preset_payload, f, ensure_ascii=False, indent=2)
                    
            logger.info("Конфигурация пресетов успешно сохранена.")
        except Exception as e:
            logger.error(f"Не удалось сохранить конфигурацию: {e}")

    @staticmethod
    def get_monaco_pretty_name(organ_name: str) -> str:
        """
        Возвращает красивое имя OAR, совместимое с Elekta Monaco 5.51 и интерфейсом.
        """
        if organ_name in MONACO_NAMES_MAP:
            return MONACO_NAMES_MAP[organ_name]
            
        pretty = organ_name
        if pretty.endswith("_left"):
            pretty = pretty[:-5] + "_l"
        elif pretty.endswith("_right"):
            pretty = pretty[:-6] + "_r"
            
        pretty = pretty.replace("_", " ").title()
        return pretty

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
    def get_all_supported_organs() -> List[str]:
        """Динамически получает список всех органов из TotalSegmentator."""
        try:
            from totalsegmentator.map_to_binary import class_map
            supported = set()
            allowed_tasks = ['total', 'total_v1', 'brain_structures', 'head_glands_cavities', 'face']
            for task in allowed_tasks:
                if task in class_map:
                    supported.update(class_map[task].values())
            return sorted(list(supported))
        except Exception as e:
            logger.warning(f"Не удалось получить список органов: {e}")
            return []

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
        return (labeled_array == largest_label).astype(bool)

    @staticmethod
    def smooth_3d_mask(mask_3d: np.ndarray, sigma: float) -> np.ndarray:
        """
        Постобработка: сглаживает ступенчатость контуров 3D-маски с помощью Гауссова фильтра.
        """
        if sigma <= 0.0:
            return mask_3d
        smoothed = gaussian_filter(mask_3d.astype(float), sigma=sigma)
        return (smoothed > 0.5).astype(bool)

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
        merge_mode: str = "merge",
        existing_rtstruct_path: Optional[str] = None,
        use_gpu: bool = False,
        remove_blobs: bool = False,
        smoothing_sigma: float = 0.0,
        step_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        is_cancelled_cb: Optional[Callable[[], bool]] = None,
        register_process_cb: Optional[Callable[[subprocess.Popen], None]] = None
    ) -> Tuple[int, float]:
        """
        Основной пайплайн выполнения автооконтурирования органов риска на КТ.
        Возвращает кортеж: (количество добавленных структур, время выполнения в секундах).
        """
        start_time = time.time()
        logger.info(f"[DIAGNOSTIC] run_pipeline вызвана. step_callback={step_callback}, progress_callback={progress_callback}")
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
            
            patient_id = "Unknown"
            patient_name = "Unknown"
            study_date = "Unknown"
            series_uid = "Unknown"
            try:
                dicom_files = list(dicom_dir.glob("*.dcm")) + list(dicom_dir.glob("*.DCM"))
                if not dicom_files:
                    dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
                
                for dcm_path in dicom_files:
                    try:
                        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                        patient_id = getattr(ds, "PatientID", "Unknown")
                        series_uid = getattr(ds, "SeriesInstanceUID", "Unknown")
                        raw_name = getattr(ds, "PatientName", "")
                        patient_name = str(raw_name).replace("^", " ").strip() if raw_name else "Unknown"
                        study_date = getattr(ds, "StudyDate", "Unknown")
                        
                        if patient_id != "Unknown" or patient_name != "Unknown":
                            logger.info(f"Успешно считаны метаданные: {patient_name}, {patient_id}, {study_date}")
                            break
                    except Exception:
                        continue
            except Exception as de:
                logger.debug(f"Не удалось получить список DICOM файлов для метаданных: {de}")

            # ----------------------------------------------------------------------
            # Шаг 1: Конвертация DICOM -> NIfTI
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 1 из 5: Конвертация DICOM в NIfTI...")
            if progress_callback:
                progress_callback(2, "Шаг 1/5: Сборка 3D-тома NIfTI...")
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
                step_callback("Шаг 2 из 5: ИИ сегментирует органы...")
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
            
            # ---- АДАПТАЦИЯ И РАЗДЕЛЕНИЕ ОРГАНОВ ПО ЗАДАЧАМ ----
            # Получаем список органов только базовой модели (total) — для валидации фолбэков
            total_supported = set()
            try:
                from totalsegmentator.map_to_binary import class_map
                if "total" in class_map:
                    total_supported = set(class_map["total"].values())
                if "total_v1" in class_map:
                    total_supported.update(class_map["total_v1"].values())
                logger.info(f"Динамически загружено классов total TotalSegmentator: {len(total_supported)}")
            except Exception as e:
                logger.warning(f"Не удалось динамически получить список классов TotalSegmentator: {e}. Резервный набор.")
                total_supported = set(self.colors.keys())

            # Карта виртуальных органов, которые мы можем собрать из долей/частей
            VIRTUAL_ORGANS_MAP = {
                "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
                "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
                "lungs": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
                "brain_stem": ["brainstem"]
            }

            # "body" — специальный орган: требует отдельного вызова
            need_body_task = target_organs is not None and "body" in target_organs
            if need_body_task:
                logger.info("Contour 'body' запрошен: будет выполнен программный расчет контура тела без ИИ.")

            # Находим путь к исполняемому файлу TotalSegmentator в виртуальном окружении
            exe_dir = Path(sys.executable).parent
            totalseg_exe = exe_dir / "TotalSegmentator.exe"
            if not totalseg_exe.exists():
                totalseg_exe = exe_dir / "TotalSegmentator"
            if not totalseg_exe.exists():
                totalseg_exe = Path("TotalSegmentator")

            # ---- ДИНАМИЧЕСКОЕ РАЗДЕЛЕНИЕ ПО ЗАДАЧАМ ----
            tasks_to_run = {}
            skipped_organs = []

            # Словарь маппинга наших имен OAR на имена классов в ИИ TotalSegmentator
            AI_CLASS_NAME_MAP = {
                "lens_left": "eye_lens_left",
                "lens_right": "eye_lens_right",
                "iliac_vein_left": "iliac_vena_left",
                "iliac_vein_right": "iliac_vena_right",
                "brain_stem": "brainstem",
                "thalamus_left": "thalamus",
                "thalamus_right": "thalamus",
                "hippocampus_left": "hippocampus",
                "hippocampus_right": "hippocampus",
                "amygdala_left": "amygdala",
                "amygdala_right": "amygdala",
                "caudate_left": "caudate_nucleus",
                "caudate_right": "caudate_nucleus",
                "putamen_left": "lentiform_nucleus",
                "putamen_right": "lentiform_nucleus",
                "pallidum_left": "lentiform_nucleus",
                "pallidum_right": "lentiform_nucleus",
            }

            try:
                from totalsegmentator.map_to_binary import class_map
            except Exception as e:
                logger.warning(f"Не удалось загрузить class_map из TotalSegmentator: {e}")
                class_map = {}

            if precision_mode == "faster":
                tasks_to_run["body"] = []
            elif target_organs:
                for organ in target_organs:
                    if organ == "body":
                        continue

                    # Если это виртуальный орган (например, lung_left или brain_stem), разбиваем его на части
                    if organ in VIRTUAL_ORGANS_MAP:
                        parts = VIRTUAL_ORGANS_MAP[organ]
                        logger.info(f"Орган '{organ}' разбит на составные части: {parts}")
                        resolved_parts = parts
                    else:
                        resolved_parts = [organ]

                    for part in resolved_parts:
                        # Маппим наше имя на имя в ИИ
                        ai_part = AI_CLASS_NAME_MAP.get(part, part)

                        # Изначально берем задачу из конфига, иначе 'total'
                        task = ROI_TO_TASK_MAP.get(part, 'total')

                        # Проверяем реальную доступность класса в этой задаче
                        is_valid = False
                        if class_map:
                            if task in class_map and ai_part in class_map[task].values():
                                is_valid = True
                            else:
                                # Класс отсутствует в назначенной задаче.
                                # Ищем, в какой задаче в class_map он реально есть.
                                found_task = None
                                for t_name, t_classes in class_map.items():
                                    if ai_part in t_classes.values():
                                        found_task = t_name
                                        break
                                
                                if found_task:
                                    logger.info(f"Орган '{part}' (класс ИИ '{ai_part}') перенаправлен из задачи '{task}' в задачу '{found_task}'.")
                                    task = found_task
                                    is_valid = True
                        else:
                            # Если class_map не загрузился, доверяем статическому маппингу
                            is_valid = True

                        if is_valid:
                            if task not in tasks_to_run:
                                tasks_to_run[task] = []
                            tasks_to_run[task].append(ai_part)
                        else:
                            logger.warning(f"Орган '{part}' (класс ИИ '{ai_part}') не поддерживается текущей версией TotalSegmentator и будет пропущен.")
                            skipped_organs.append(part)

            # Дедупликация ROI внутри каждой задачи
            for task_name in tasks_to_run:
                tasks_to_run[task_name] = sorted(list(set(tasks_to_run[task_name])))

            logger.info(f"Итоговый план задач ИИ: { {t: rois for t, rois in tasks_to_run.items()} }")

            if not tasks_to_run and precision_mode != "faster":
                raise RuntimeError(
                    "Ни один из выбранных органов не поддерживается текущей версией TotalSegmentator.\n"
                    "Пожалуйста, выберите другие органы риска (например, мочевой пузырь или кости)."
                )

            # Выполняем ИИ-модели последовательно для каждой задачи
            for task_index, (task_name, task_rois) in enumerate(tasks_to_run.items(), start=1):
                if is_cancelled_cb and is_cancelled_cb():
                    raise RuntimeError("Операция отменена пользователем.")
                    
                cmd = [
                    sys.executable,
                    "-m",
                    "totalsegmentator.bin.TotalSegmentator",
                    "-i", str(nifti_ct_path),
                    "-o", str(segmentation_dir),
                    "--device", device,
                    "--nr_thr_resamp", "1",
                    "--nr_thr_saving", "1"
                ]
                
                if task_name in LICENSED_TASKS and hasattr(self, "licenses") and isinstance(self.licenses, str) and self.licenses.strip():
                    cmd.extend(["--license", self.licenses.strip()])
                
                if precision_mode == "fast" or precision_mode == "faster":
                    if task_name == "total" or task_name == "body":
                        cmd.append("--fast")
                    else:
                        logger.info(f"Для задачи {task_name} режим --fast принудительно отключен, так как суб-модели требуют высокого разрешения.")
                
                if task_name != "total":
                    cmd.extend(["--task", task_name])
                
                if task_rois and precision_mode != "faster":
                    if task_name == "total":
                        cmd.append("--roi_subset")
                        cmd.extend(task_rois)
                    else:
                        logger.info(f"Флаг --roi_subset пропущен для задачи {task_name}, так как он поддерживается только для задачи 'total'. Будут сгенерированы все органы суб-модели (лишнее отсеется при сборке).")
                logger.info(f"Запуск внешнего процесса TotalSegmentator (Task: {task_name}): {' '.join(cmd)}")
                
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
                    
                current_loop_index = 0
                last_percent = 0
                total_tasks = len(tasks_to_run)
                    
                import queue
                from threading import Thread

                def enqueue_output(out, queue_obj):
                    try:
                        for line in iter(out.readline, ''):
                            queue_obj.put(line)
                        out.close()
                    except Exception:
                        pass

                q = queue.Queue()
                t = Thread(target=enqueue_output, args=(process.stdout, q))
                t.daemon = True
                t.start()

                while True:
                    if is_cancelled_cb and is_cancelled_cb():
                        if process.poll() is None:
                            logger.info(f"Отмена: Принудительное завершение задачи {task_name}...")
                            process.kill()
                        raise RuntimeError("Операция отменена пользователем.")
                        
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        line = None
                        
                    if line is None and process.poll() is not None:
                        if q.empty():
                            break
                        else:
                            time.sleep(0.02)
                            continue
                            
                    if line:
                        clean_line = line.strip()
                        if clean_line:
                            logger.info(f"[TotalSegmentator {task_name}]: {clean_line}")
                            if step_callback is not None:
                                try:
                                    step_callback(f"[TotalSegmentator {task_name}]: {clean_line}")
                                except Exception as cb_err:
                                    logger.error(f"Ошибка вызова step_callback: {cb_err}")
                            else:
                                logger.warning(f"step_callback является None во время логирования TotalSegmentator {task_name}")
                            
                            if progress_callback:
                                match = re.search(r'(\d+)%\|', clean_line)
                                if match:
                                    sub_percent = int(match.group(1))
                                    if sub_percent == 0 and last_percent == 100:
                                        current_loop_index += 1
                                    last_percent = sub_percent
                                    
                                    global_ai_percent = 5 + int((((task_index - 1) * 500) + (current_loop_index * 100) + sub_percent) / (total_tasks * 500) * 90)
                                    if current_loop_index == 0:
                                        txt = f"Шаг 2/5: ИИ определяет границы тела (Задача {task_index}/{total_tasks}: {task_name})..."
                                    else:
                                        txt = f"Шаг 2/5: ИИ оконтуривает структуры (Задача {task_index}/{total_tasks}: {task_name}). Расчет части {current_loop_index} из 4..."
                                    progress_callback(global_ai_percent, txt)
                            
                return_code = process.wait()
                if return_code != 0:
                    logger.error(
                        f"[ERROR]: Процесс TotalSegmentator для задачи '{task_name}' завершился с кодом ошибки {return_code}.\n"
                        f"Возможная причина: отсутствие академической/коммерческой лицензии для этой суб-модели "
                        f"(например, для 'brain_structures') или отсутствие установленного веса.\n"
                        f"Пайплайн продолжит работу для остальных задач."
                    )
                    continue
                
            logger.info(f"Шаг 2 успешно завершен за {time.time() - step_start:.2f} сек.")

            # ------------------------------------------------------------------
            # Опционально: запуск TotalSegmentator --task body для контура тела
            # ------------------------------------------------------------------
            if need_body_task and precision_mode != "faster":
                if is_cancelled_cb and is_cancelled_cb():
                    raise RuntimeError("Операция отменена пользователем.")
                logger.info("--- Доп. задача: Программный расчет контура тела без ИИ (Fast Classic Skin) ---")
                try:
                    import scipy.ndimage
                    if nifti_ct_path.exists():
                        logger.info("Чтение КТ-объема для выделения тела...")
                        ct_nii = nib.load(str(nifti_ct_path))
                        ct_data = ct_nii.get_fdata()
                        
                        # Порог для тела: воздух на КТ равен -1000 HU, тело обычно > -200 HU
                        logger.info("Применение пороговой фильтрации (-200 HU)...")
                        body_mask = ct_data > -200
                        
                        # Заливаем полости на каждом срезе
                        logger.info("Заливка внутренних полостей тела (fill holes)...")
                        filled_mask = np.zeros_like(body_mask, dtype=bool)
                        for z in range(body_mask.shape[2]):
                            filled_mask[:, :, z] = scipy.ndimage.binary_fill_holes(body_mask[:, :, z])
                            
                        # Оставляем только крупнейший связный компонент (чтобы убрать кушетку, воздух вокруг и артефакты)
                        logger.info("Выделение крупнейшего 3D связного компонента...")
                        labeled_array, num_features = label(filled_mask)
                        if num_features > 1:
                            sizes = np.bincount(labeled_array.ravel())
                            largest_label = np.argmax(sizes[1:]) + 1
                            final_mask = labeled_array == largest_label
                        else:
                            final_mask = filled_mask
                            
                        # Сохраняем маску тела в папку масок
                        body_nii_img = nib.Nifti1Image(final_mask.astype(np.uint8), ct_nii.affine, ct_nii.header)
                        dest = segmentation_dir / "body.nii.gz"
                        nib.save(body_nii_img, str(dest))
                        logger.info(f"Контур тела body.nii.gz успешно сгенерирован программно и сохранен в: {dest.name}")
                    else:
                        logger.warning("Не найден временный NIfTI-файл КТ для программного расчета тела.")
                except Exception as body_err:
                    logger.error(f"Ошибка при программном расчете контура тела: {body_err}", exc_info=True)

            # ----------------------------------------------------------------------
            # Постобработка: сборка виртуальных органов (легкие, ствол мозга)
            # ----------------------------------------------------------------------
            logger.info("--- Постобработка: Сборка цельных легких и ствола мозга из частей ---")
            try:
                POST_VIRTUAL_MAP = {
                    "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
                    "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
                    "brain_stem": ["brainstem"]
                }
                
                for virtual_organ, parts in POST_VIRTUAL_MAP.items():
                    part_files = [segmentation_dir / f"{part}.nii.gz" for part in parts]
                    existing_part_files = [f for f in part_files if f.exists()]
                    
                    if existing_part_files:
                        logger.info(f"Сборка цельного органа '{virtual_organ}' из частей: {[f.name for f in existing_part_files]}")
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
                            if part_file.name != f"{virtual_organ}.nii.gz":
                                try:
                                    part_file.unlink()
                                except Exception as e:
                                    logger.debug(f"Не удалось удалить файл части {part_file.name}: {e}")
            except Exception as e:
                logger.error(f"Не удалось завершить сборку виртуальных органов: {e}")

            # --------------------------------------------------------------
            # Постобработка: расщепление комбинированных структур мозга на Left/Right
            # --------------------------------------------------------------
            logger.info("--- Постобработка: Расщепление комбинированных структур мозга на L/R ---")
            try:
                SPLIT_MAP = {
                    "thalamus": ["thalamus_left", "thalamus_right"],
                    "caudate_nucleus": ["caudate_left", "caudate_right"],
                    "lentiform_nucleus": ["putamen_left", "putamen_right", "pallidum_left", "pallidum_right"]
                }
                
                for combined_org, target_parts in SPLIT_MAP.items():
                    combined_file = segmentation_dir / f"{combined_org}.nii.gz"
                    if combined_file.exists():
                        logger.info(f"Обнаружена комбинированная маска '{combined_org}', расщепление на {target_parts}...")
                        nii = nib.load(str(combined_file))
                        data = nii.get_fdata()
                        affine = nii.affine
                        header = nii.header
                        
                        width = data.shape[0]
                        mid = width // 2
                        
                        for part in target_parts:
                            part_data = np.zeros_like(data)
                            if "left" in part or "l" in part.split("_")[-1]:
                                part_data[:mid, :, :] = data[:mid, :, :]
                            else:
                                part_data[mid:, :, :] = data[mid:, :, :]
                                
                            if np.any(part_data > 0.5):
                                part_nii = nib.Nifti1Image(part_data.astype(np.uint8), affine, header)
                                part_file_path = segmentation_dir / f"{part}.nii.gz"
                                nib.save(part_nii, str(part_file_path))
                                logger.info(f"  Успешно создана маска полушария: {part_file_path.name}")
                            else:
                                logger.info(f"  Маска {part} пуста после расщепления, пропускаем.")
                                
                        combined_file.unlink()
            except Exception as split_err:
                logger.error(f"Ошибка при расщеплении комбинированных структур: {split_err}")

            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")

            # ----------------------------------------------------------------------
            # Шаг 3: Очистка временных файлов
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 3 из 5: Фильтрация артефактов и сглаживание масок...")
            logger.info("--- Шаг 3 из 5: Очистка ОЗУ перед фильтрацией ---")
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
                step_callback("Шаг 4 из 5: Формирование DICOM RTSTRUCT и привязка геометрии...")
            logger.info("--- Шаг 4 из 5: Формирование DICOM RTSTRUCT и привязка к геометрии ---")
            
            step_start = time.time()
            
            # ----------------------------------------------------------------------
            # Программное объединение левого и правого легкого в общий контур Lungs
            # ----------------------------------------------------------------------
            if precision_mode != "faster" and (target_organs is None or "lungs" in target_organs):
                left_nii = segmentation_dir / "lung_left.nii.gz"
                right_nii = segmentation_dir / "lung_right.nii.gz"
                if left_nii.exists() and right_nii.exists():
                    try:
                        logger.info("Программное объединение левого и правого легкого в общий контур Lungs...")
                        left_img = nib.load(str(left_nii))
                        right_img = nib.load(str(right_nii))
                        
                        merged_data = (left_img.get_fdata() > 0.5) | (right_img.get_fdata() > 0.5)
                        
                        lungs_img = nib.Nifti1Image(merged_data.astype(np.uint8), left_img.affine, left_img.header)
                        lungs_dest = segmentation_dir / "lungs.nii.gz"
                        nib.save(lungs_img, str(lungs_dest))
                        logger.info(f"Общий контур легких успешно сгенерирован и сохранен как: {lungs_dest.name}")
                    except Exception as merge_err:
                        logger.error(f"Не удалось объединить контуры легких в общий Lungs: {merge_err}", exc_info=True)

            mask_files = list(segmentation_dir.glob("*.nii.gz"))
            if not mask_files:
                raise RuntimeError("Не найдено масок органов после сегментации.")
                
            detected_organs = sorted([FILE_NAME_MAP.get(f.name.replace(".nii.gz", ""), f.name.replace(".nii.gz", "")) for f in mask_files])
            logger.info(f"Обнаружено сегментированных масок органов: {len(mask_files)}")
            logger.info(f"Список определенных ИИ органов на КТ: {detected_organs}")
            
            existing_rois = []
            rtstruct = None
            
            if merge_mode == "merge" and existing_rtstruct_path:
                rt_path = Path(existing_rtstruct_path)
                if rt_path.exists():
                    try:
                        logger.info(f"Загрузка существующего RTSTRUCT для автоматического слияния: {rt_path}")
                        rtstruct = RTStructBuilder.create_from(
                            dicom_series_path=str(dicom_dir),
                            rt_struct_path=str(rt_path),
                            warn_only=True
                        )
                        existing_rois = rtstruct.get_roi_names()
                        logger.info(f"Существующие структуры в файле: {existing_rois}")
                        
                        # Создаем бэкап оригинального файла
                        backup_path = rt_path.with_name(rt_path.name + ".backup")
                        shutil.copy2(str(rt_path), str(backup_path))
                        logger.info(f"Создан бэкап старого файла структур: {backup_path}")
                        
                    except Exception as e:
                        logger.error(
                            f"Не удалось загрузить RTSTRUCT '{rt_path}' для слияния: {e}. "
                            "Пайплайн переключен в режим создания НОВОГО файла."
                        )
                        existing_rtstruct_path = None
                else:
                    logger.error(
                        f"Файл RTSTRUCT для слияния не найден на диске: '{rt_path}'. "
                        "Пайплайн переключен в режим создания НОВОГО файла."
                    )
                    existing_rtstruct_path = None
                    
            if rtstruct is None:
                # В режимах "new" или "overwrite" (или если "merge" не удался), создаем чистый RTSTRUCT
                logger.info("Создание НОВОГО файла RTSTRUCT (без слияния)...")
                rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dicom_dir))
                existing_rtstruct_path = None

            # Копирование критичных DICOM тегов для совместимости с Elekta Monaco
            try:
                import pydicom
                first_dcm_path = next(Path(dicom_dir).glob("*.dcm"))
                ref_dcm = pydicom.dcmread(str(first_dcm_path), stop_before_pixels=True)
                if hasattr(ref_dcm, 'FrameOfReferenceUID'):
                    rtstruct.ds.FrameOfReferenceUID = ref_dcm.FrameOfReferenceUID
                if hasattr(ref_dcm, 'PositionReferenceIndicator'):
                    rtstruct.ds.PositionReferenceIndicator = ref_dcm.PositionReferenceIndicator
                if hasattr(ref_dcm, 'PatientPosition'):
                    rtstruct.ds.PatientPosition = ref_dcm.PatientPosition
                logger.info("Скопированы DICOM теги (FrameOfReferenceUID, PositionReferenceIndicator, PatientPosition) для совместимости с TPS Monaco.")
            except StopIteration:
                pass
            except Exception as e:
                logger.warning(f"Не удалось скопировать DICOM теги для Monaco: {e}")
            
            added_count = 0
            
            # Извлекаем алиасы для дублирования контуров из пресета
            organ_to_aliases = {}
            preset_items = self.presets.get(preset_name, [])
            for item in preset_items:
                if isinstance(item, dict):
                    for k, v in item.items():
                        organ_to_aliases[k] = v
            
            # Обратный маппинг классов ИИ на наши стандартные имена OAR (для дублирования/разделения объединенных классов)
            AI_TO_STANDARD_MAP = {
                "eye_lens_left": ["lens_left"],
                "eye_lens_right": ["lens_right"],
                "iliac_vena_left": ["ilia_vein_left", "iliac_vein_left"],  # на всякий случай опечатку тоже
                "iliac_vena_right": ["iliac_vein_right"],
                "brainstem": ["brain_stem"],
                "thalamus": ["thalamus_left", "thalamus_right"],
                "hippocampus": ["hippocampus_left", "hippocampus_right"],
                "amygdala": ["amygdala_left", "amygdala_right"],
                "caudate_nucleus": ["caudate_left", "caudate_right"],
                "lentiform_nucleus": ["putamen_left", "putamen_right", "pallidum_left", "pallidum_right"],
            }

            resolved_mask_items = []
            for mask_file in mask_files:
                raw_name = mask_file.name.replace(".nii.gz", "")
                if raw_name in AI_TO_STANDARD_MAP:
                    for std_name in AI_TO_STANDARD_MAP[raw_name]:
                        resolved_mask_items.append((mask_file, std_name))
                else:
                    std_name = FILE_NAME_MAP.get(raw_name, raw_name)
                    resolved_mask_items.append((mask_file, std_name))

            for idx, (mask_file, organ_name) in enumerate(resolved_mask_items):
                if progress_callback:
                    prog = 95 + int((idx / len(resolved_mask_items)) * 5)
                    progress_callback(prog, f"Шаг 4/5: Сглаживание и фильтрация ROI {organ_name}...")
                
                # Фильтруем по списку целевых органов (если это не сверхбыстрый режим body, где ищется всё тело)
                if precision_mode != "faster" and target_organs and organ_name not in target_organs:
                    continue
                    
                logger.info(f"Обработка органа: {organ_name}...")
                
                try:
                    nii_mask = nib.load(str(mask_file))
                    mask_data = nii_mask.get_fdata() > 0.5
                    
                    # ------------------------------------------------------------------
                    # ПОСТОБРАБОТКА МАСОК (Remove small blobs & Smoothing)
                    # ------------------------------------------------------------------
                    # Органы, для которых нельзя применять remove_small_blobs:
                    # 1) Очень мелкие структуры (линзы, нервы) — могут быть целиком удалены
                    # 2) Парные структуры в одной маске (лёгкие, доли мозга) — 
                    #    содержат 2+ отдельных 3D-компонента, remove_blobs удалит один из них
                    SKIP_BLOB_REMOVAL = {
                        "lens_left", "lens_right", "eye_left", "eye_right",
                        "optic_nerve_left", "optic_nerve_right",
                        "lungs",  # объединённый контур обоих лёгких
                        "temporal_lobe", "frontal_lobe", "parietal_lobe",
                        "occipital_lobe", "insular_cortex",  # доли мозга обоих полушарий
                    }
                    
                    if remove_blobs and organ_name not in SKIP_BLOB_REMOVAL:
                        before_pixels = np.sum(mask_data)
                        mask_data = self.remove_small_blobs(mask_data)
                        after_pixels = np.sum(mask_data)
                        removed_pixels = before_pixels - after_pixels
                        if removed_pixels > 0:
                            logger.info(f"[{organ_name}] Удалено мелких артефактов (blobs): {removed_pixels} пикселей")

                    # Мелкие органы не сглаживаем — Гаусс может "размыть" их до пустоты
                    SKIP_SMOOTHING = {"lens_left", "lens_right", "eye_left", "eye_right", "optic_nerve_left", "optic_nerve_right"}
                    if smoothing_sigma > 0.0 and organ_name not in SKIP_SMOOTHING:
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
                    
                    if organ_name in organ_to_aliases:
                        aliases = organ_to_aliases[organ_name]
                        for i, alias in enumerate(aliases):
                            # Слегка меняем цвет дубликатов для визуального отличия
                            adj_color = color.copy()
                            if i > 0:
                                adj_color = [adj_color[0], min(255, adj_color[1] + 40 * i), adj_color[2]]
                            roi_names_to_add.append((alias, adj_color))
                    else:
                        # Стандартное английское название, совместимое с Elekta Monaco 5.51
                        pretty_name = self.get_monaco_pretty_name(organ_name)
                        roi_names_to_add.append((pretty_name, color))
                    
                    # Добавление в RTSTRUCT
                    for roi_name, roi_color in roi_names_to_add:
                        if merge_mode == "merge" and roi_name in existing_rois:
                            logger.info(f"Структура {roi_name} уже существует в исходном файле, пропускаем обработку ИИ.")
                            continue
                        
                        rtstruct.add_roi(
                            mask=mask_bool,
                            color=roi_color,
                            name=roi_name
                        )
                        added_count += 1
                        logger.info(f"Успешно добавлен ROI '{roi_name}' (цвет: {roi_color})")
                    
                    # Очистка памяти после обработки маски органа
                    del mask_data
                    del mask_data_transposed
                    del mask_bool
                    gc.collect()
                except Exception as organ_err:
                    logger.error(f"Ошибка при обработке органа {organ_name}: {organ_err}", exc_info=True)
                    # Гарантируем очистку памяти в случае частичной ошибки
                    if 'mask_data' in locals():
                        try:
                            del mask_data
                        except NameError:
                            pass
                    if 'mask_data_transposed' in locals():
                        try:
                            del mask_data_transposed
                        except NameError:
                            pass
                    if 'mask_bool' in locals():
                        try:
                            del mask_bool
                        except NameError:
                            pass
                    gc.collect()
                
            if added_count == 0:
                if merge_mode == "merge" and existing_rois:
                    logger.info("В режиме слияния не добавлено новых ИИ-структур, так как они уже присутствуют в файле. Сохраняем исходный файл.")
                else:
                    raise RuntimeError("В RTSTRUCT не было добавлено ни одного OAR. Проверьте область сканирования.")
                
            if is_cancelled_cb and is_cancelled_cb():
                raise RuntimeError("Операция отменена пользователем.")

            # ----------------------------------------------------------------------
            # Шаг 5: Сохранение итогового файла
            # ----------------------------------------------------------------------
            if step_callback:
                step_callback("Шаг 5 из 5: Успешно сохранено!")
            logger.info("--- Шаг 5 из 5: Запись итогового DICOM RTSTRUCT ---")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            clean_patient_id = "".join([c for c in str(patient_id) if c.isalnum() or c in ("_", "-")]).strip()
            if not clean_patient_id:
                clean_patient_id = "Unknown"
                
            clean_patient_name = "".join([c if c.isalnum() else "_" for c in str(patient_name)]).strip("_")
            if not clean_patient_name:
                clean_patient_name = "Unknown"
                
            clean_study_date = "".join([c for c in str(study_date) if c.isalnum()]).strip()
            if not clean_study_date:
                clean_study_date = "Unknown"

            if existing_rtstruct_path and merge_mode in ["merge", "overwrite"]:
                # Берем исходное имя существующего файла структур без изменений
                rtstruct_filename = Path(existing_rtstruct_path).name
            else:
                # Упрощенное именование по таймстампу, как попросил пользователь
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rtstruct_filename = f"STR_{timestamp}.dcm"

            rtstruct_file_path = output_dir / rtstruct_filename
            
            rtstruct.save(str(rtstruct_file_path))
            logger.info(f"Шаг 5 успешно завершен за {time.time() - step_start:.2f} сек.")
            elapsed_total = time.time() - start_time
            logger.info(f"Итоговый файл RTSTRUCT успешно записан: {rtstruct_file_path}")
            
            # Очистка временной папки
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.debug(f"Не удалось удалить временную папку {temp_dir}: {e}")
                
            final_count = len(rtstruct.get_roi_names())
            return final_count, elapsed_total
            
        except Exception as e:
            logger.error(f"Сбой в пайплайне: {e}", exc_info=True)
            logger.warning(f"Временная рабочая папка сохранена: {temp_dir}")
            raise e
            
        else:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
        finally:
            # Очистка ссылок на все тяжелые массивы данных во избежание утечек памяти
            ct_nii = None
            ct_data = None
            body_mask = None
            filled_mask = None
            labeled_array = None
            final_mask = None
            body_nii_img = None
            base_nii = None
            base_data = None
            part_data = None
            merged_nii = None
            nii = None
            data = None
            part_nii = None
            rtstruct = None
            ref_dcm = None
            
            gc.collect()
            
            logger.info(f"Пайплайн завершен. Общее время работы: {time.time() - start_time:.2f} сек. Ресурсы памяти очищены.")
