#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Скрипт автооконтурирования органов риска (OAR) на КТ: Графический интерфейс PyQt6
================================================================================
Этот файл содержит графическую оболочку приложения AI Contour на PyQt6.
Вся тяжелая вычислительная логика вынесена в модуль contour_engine.py.

Особенности:
1. Разделение настроек на вкладки для эргономики.
2. Динамическое управление пресетами и цветами из конфигурации config/.
3. Интерактивная кастомизация цветов органов прямо в списке через QColorDialog.
4. Выбор вычислительных ресурсов (CPU/GPU) и режимов точности TotalSegmentator.
5. Интегрированные чекбоксы 3D постобработки (Blobs / Сглаживание).
================================================================================
"""

import os
import sys
import gc
import shutil
import time
import math
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import pyqtgraph as pg
import numpy as np
import pydicom

# Импорт PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QListWidget, QListWidgetItem,
        QRadioButton, QButtonGroup, QTextEdit, QProgressBar, QFileDialog,
        QMessageBox, QFrame, QSplitter, QCheckBox, QDialog, QTextBrowser,
        QTabWidget, QColorDialog, QGroupBox,
        QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMenu,
        QProgressDialog, QScrollArea, QGridLayout
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QSettings, QTimer
    from PyQt6.QtGui import QTextCursor, QBrush, QColor, QFont, QIcon, QPixmap, QPalette
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Импортируем вычислительный движок или используем легковесный заглушечный класс для клиента
try:
    from contour_engine import ContourEngine
except ImportError:
    ContourEngine = None

if ContourEngine is None:
    class MockContourEngine:
        """Легковесный клиентский класс для загрузки конфигурации без тяжелых ML библиотек."""
        def __init__(self, *args, **kwargs):
            self.ru_names = {}
            self.colors = {}
            self.presets = {}
            self.licenses = ""
            self.load_presets_config()
            
        def is_gpu_available(self):
            # Клиент не проводит вычислений, GPU всегда выключено для локальных тестов
            return False
            
        def load_presets_config(self):
            import json
            import sys
            from pathlib import Path
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent.resolve()
            config_dir = (base_dir / "config").resolve()
            colors_path = config_dir / "colors.json"
            translations_path = config_dir / "translations.json"
            presets_dir = config_dir / "presets"
            licenses_path = config_dir / "licenses.json"
            
            try:
                if colors_path.exists():
                    with open(colors_path, "r", encoding="utf-8") as f:
                        self.colors = json.load(f)
                if translations_path.exists():
                    with open(translations_path, "r", encoding="utf-8") as f:
                        self.ru_names = json.load(f)
                if presets_dir.exists():
                    for p_file in presets_dir.glob("*.json"):
                        with open(p_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            name = data.get("name")
                            organs = data.get("organs", [])
                            if name:
                                self.presets[name] = organs
                if licenses_path.exists():
                    with open(licenses_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.licenses = data.get("license_key", "").strip()
                else:
                    self.licenses = ""
            except Exception as e:
                print(f"Ошибка загрузки локальных конфигураций на клиенте: {e}")
                
        def save_presets_config(self):
            import json
            import sys
            from pathlib import Path
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent.resolve()
            config_dir = (base_dir / "config").resolve()
            config_dir.mkdir(parents=True, exist_ok=True)
            
            colors_path = config_dir / "colors.json"
            translations_path = config_dir / "translations.json"
            licenses_path = config_dir / "licenses.json"
            
            try:
                with open(colors_path, "w", encoding="utf-8") as f:
                    json.dump(self.colors, f, ensure_ascii=False, indent=2)
                with open(translations_path, "w", encoding="utf-8") as f:
                    json.dump(self.ru_names, f, ensure_ascii=False, indent=2)
                with open(licenses_path, "w", encoding="utf-8") as f:
                    json.dump({"license_key": getattr(self, "licenses", "")}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Ошибка сохранения локальных конфигураций на клиенте: {e}")
                
        def get_all_supported_organs(self):
            """Возвращает список всех поддерживаемых органов из загруженных переводов."""
            return sorted(list(self.ru_names.keys()))

        def get_monaco_pretty_name(self, organ_name: str) -> str:
            """Возвращает красивое имя OAR, совместимое с Elekta Monaco 5.51 и интерфейсом."""
            from config import MONACO_NAMES_MAP
            if organ_name in MONACO_NAMES_MAP:
                return MONACO_NAMES_MAP[organ_name]
                
            pretty = organ_name
            if pretty.endswith("_left"):
                pretty = pretty[:-5] + "_l"
            elif pretty.endswith("_right"):
                pretty = pretty[:-6] + "_r"
                
            pretty = pretty.replace("_", " ").title()
            return pretty
            
        def _get_default_color(self, organ_name: str):
            import hashlib
            h = hashlib.md5(organ_name.encode('utf-8')).digest()
            return [max(50, int(h[0])), max(50, int(h[1])), max(50, int(h[2]))]
            
    ContourEngine = MockContourEngine

# Импортируем конфигурационные данные
try:
    from config import ORGAN_GROUPS, EXTERNAL_ALIASES, LICENSED_TASKS, ROI_TO_TASK_MAP, StatisticsManager
except ImportError:
    ORGAN_GROUPS = {}
    EXTERNAL_ALIASES = {}
    LICENSED_TASKS = {}
    ROI_TO_TASK_MAP = {}
    class StatisticsManager:
        def __init__(self, *args, **kwargs): pass
        def get_stats(self):
            return {
                "total_runs": 0, "successful_runs": 0, "failed_runs": 0, "cancelled_runs": 0,
                "total_organs_contoured": 0, "total_elapsed_time_seconds": 0.0,
                "organ_stats": {}, "recent_runs": []
            }
        def record_run(self, *args, **kwargs): pass
        def reset_stats(self): pass

# Настройка логирования с автоматической ротацией
from logging.handlers import RotatingFileHandler

# Создаем папку для логов в корне проекта
try:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
except Exception as e:
    print(f"Ошибка создания папки логов: {e}")

log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Консольный вывод логов
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s]: %(message)s', datefmt='%H:%M:%S'))
stream_handler.setLevel(logging.INFO)

# Ротационный обработчик основного лога (до 2 файлов по 5 МБ)
try:
    general_handler = RotatingFileHandler(
        Path("logs/auto_contour.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8"
    )
    general_handler.setFormatter(log_formatter)
    general_handler.setLevel(logging.INFO)
except Exception as e:
    general_handler = None
    print(f"Ошибка инициализации основного лог-файла: {e}")

# Ротационный обработчик лога ошибок (до 2 файлов по 5 МБ)
try:
    error_handler = RotatingFileHandler(
        Path("logs/auto_contour_error.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8"
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)
except Exception as e:
    error_handler = None
    print(f"Ошибка инициализации лог-файла ошибок: {e}")

# Конфигурация корневого логера
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = []  # Сброс стандартных обработчиков
root_logger.addHandler(stream_handler)
if general_handler:
    root_logger.addHandler(general_handler)
if error_handler:
    root_logger.addHandler(error_handler)

logger = logging.getLogger("AutoContourGUI")


if PYQT_AVAILABLE:
    class LogSignaler(QObject):
        """Вспомогательный класс сигналов для потокобезопасного вывода логов."""
        log_signal = pyqtSignal(str, str)

    class PauseSignaler(QObject):
        """Вспомогательный класс сигналов для потокобезопасного изменения паузы."""
        pause_signal = pyqtSignal(bool, bool)

    class ServerStatusSignaler(QObject):
        """Вспомогательный класс сигналов для потокобезопасного обновления UI сервера."""
        status_signal = pyqtSignal(bool, dict, list)

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
                                color = "#ff6b6b"
                            elif "WARNING" in line:
                                color = "#f1c40f"
                            elif "Шаг" in line or "---" in line:
                                color = "#007acc"
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

    class ViewerEventFilter(QObject):
        """Фильтр событий для переопределения скролла в pyqtgraph ImageView.
        
        - Колесико без Ctrl: прокрутка срезов.
        - Ctrl + колесико: зум (pyqtgraph ViewBox обрабатывает самостоятельно).
        """
        def __init__(self, viewer):
            super().__init__()
            self.viewer = viewer

        def eventFilter(self, obj, event):
            if event.type() == event.Type.Wheel:
                from PyQt6.QtWidgets import QApplication
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.KeyboardModifier.ControlModifier:
                    # Ctrl+колесико: передаем событие дальше в pyqtgraph (ViewBox сделает зум)
                    return False
                else:
                    # Без Ctrl: прокручиваем срезы, блокируем событие для pyqtgraph
                    delta = event.angleDelta().y()
                    if delta != 0 and self.viewer.image is not None:
                        current_idx = self.viewer.currentIndex
                        max_idx = self.viewer.image.shape[0] - 1
                        # Прокрутка от себя (delta > 0) -> к голове (увеличение Z), к себе -> от головы (уменьшение Z)
                        new_idx = min(max_idx, current_idx + 1) if delta > 0 else max(0, current_idx - 1)
                        if new_idx != current_idx:
                            self.viewer.setCurrentIndex(new_idx)
                    return True  # блокируем, чтобы pyqtgraph не зумировал
            return False

    def format_rtstruct_count(count: int) -> str:
        """Форматирует количество файлов RTSTRUCT со склонениями на русском языке."""
        if count == 0:
            return "Нет"
        remainder10 = count % 10
        remainder100 = count % 100
        if remainder100 in [11, 12, 13, 14]:
            suffix = "файлов"
        elif remainder10 == 1:
            suffix = "файл"
        elif remainder10 in [2, 3, 4]:
            suffix = "файла"
        else:
            suffix = "файлов"
        return f"{count} {suffix}"


    class DicomScanWorker(QThread):
        """Фоновый поток для сканирования папок на наличие DICOM серий."""
        scan_started = pyqtSignal(int, int, bool)  # total_dcm_count, total_dirs, is_manual
        scan_progress = pyqtSignal(int)
        scan_completed = pyqtSignal(list)
        error_signal = pyqtSignal(str)

        def __init__(self, root_dir: str, is_manual_scan: bool = True):
            super().__init__()
            self.root_dir = root_dir
            self.is_manual_scan = is_manual_scan
            self.is_cancelled = False

        def cancel(self):
            self.is_cancelled = True

        def run(self):
            import os
            import pydicom
            from pathlib import Path
            try:
                target_dirs = []
                total_dcm_count = 0
                for dirpath, dirnames, filenames in os.walk(self.root_dir):
                    if self.is_cancelled:
                        return
                    dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]
                    if dcm_files:
                        target_dirs.append((dirpath, dcm_files))
                        total_dcm_count += len(dcm_files)

                self.scan_started.emit(total_dcm_count, len(target_dirs), self.is_manual_scan)

                results = []
                for i, (dirpath, dcm_files) in enumerate(target_dirs):
                    if self.is_cancelled:
                        return
                        
                    rtstruct_count = 0
                    
                    for dcm in dcm_files:
                        if self.is_cancelled:
                            return
                        try:
                            ds = pydicom.dcmread(str(Path(dirpath) / dcm), stop_before_pixels=True)
                            if str(getattr(ds, 'Modality', '')) == 'RTSTRUCT':
                                rtstruct_count += 1
                        except Exception:
                            pass

                    slice_count = len(dcm_files) - rtstruct_count

                    if self.is_cancelled:
                        return

                    first_dcm = Path(dirpath) / dcm_files[0]
                    try:
                        ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True)
                        p_name = str(getattr(ds, 'PatientName', 'Неизвестно')).replace('^', ' ').strip()
                        p_id = str(getattr(ds, 'PatientID', 'Без ID'))
                        s_date = str(getattr(ds, 'StudyDate', ''))
                        
                        body_part = str(getattr(ds, 'BodyPartExamined', '')).strip()
                        if not body_part:
                            protocol = str(getattr(ds, 'ProtocolName', '')).lower()
                            desc = str(getattr(ds, 'SeriesDescription', '')).lower()
                            combined = protocol + " " + desc
                            if 'head' in combined: body_part = 'Head'
                            elif 'neck' in combined: body_part = 'Neck'
                            elif 'chest' in combined or 'thorax' in combined: body_part = 'Chest'
                            elif 'abdomen' in combined: body_part = 'Abdomen'
                            elif 'pelvis' in combined: body_part = 'Pelvis'
                            elif 'brachy' in combined: body_part = 'Brachytherapy'
                            else: body_part = 'Unknown'
                        
                        if len(s_date) == 8:
                            s_date = f"{s_date[6:8]}.{s_date[4:6]}.{s_date[0:4]}"
                        elif not s_date:
                            s_date = "Нет даты"
                            
                        str_status = format_rtstruct_count(rtstruct_count)
                        results.append((p_name, p_id, str_status, body_part, slice_count, s_date, dirpath))
                    except Exception:
                        pass
                    
                    if i % max(1, len(target_dirs) // 10) == 0 or i == len(target_dirs) - 1:
                        self.scan_progress.emit(i + 1)
                        
                self.scan_completed.emit(results)
            except Exception as e:
                self.error_signal.emit(str(e))

    class SegmentationWorker(QThread):
        """Поток для вычислений сегментации через удаленный API сервер, чтобы GUI не зависал."""
        finished_signal = pyqtSignal(bool, str)
        step_signal = pyqtSignal(str)
        progress_signal = pyqtSignal(int)
        eta_signal = pyqtSignal(float, float)  # (elapsed_sec, eta_sec)
        log_signal = pyqtSignal(str, str)  # (message, color)

        def __init__(
            self,
            server_url: str,
            client_name: str,
            dicom_dir: str,
            output_dir: str,
            preset_name: str,
            precision_mode: str,
            selected_organs: List[str],
            merge_mode: bool,
            existing_rtstruct_path: Optional[str],
            use_gpu: bool,
            remove_blobs: bool,
            smoothing_sigma: float
        ):
            super().__init__()
            self.server_url = server_url
            self.client_name = client_name
            self.dicom_dir = dicom_dir
            self.output_dir = output_dir
            self.preset_name = preset_name
            self.precision_mode = precision_mode
            self.selected_organs = selected_organs
            self.merge_mode = merge_mode
            self.existing_rtstruct_path = existing_rtstruct_path
            self.use_gpu = use_gpu
            self.remove_blobs = remove_blobs
            self.smoothing_sigma = smoothing_sigma
            self.is_cancelled = False
            self.job_id = None

        def cancel(self):
            self.is_cancelled = True

        def _zip_dicom_dir(self, source_dir: Path, zip_path: Path):
            """Упаковывает файлы DICOM серии в архив."""
            import zipfile
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        # Упаковываем все .dcm файлы и файлы разметки в корень
                        if file.lower().endswith('.dcm') or file.lower().endswith('.dcm.bak'):
                            file_path = Path(root) / file
                            zip_file.write(file_path, file_path.name)

        def run(self):
            import requests
            import json
            import zipfile
            import tempfile
            from pathlib import Path
            
            temp_zip = None
            job_id = None
            try:
                self.step_signal.emit("Шаг 1 из 5: Архивирование КТ снимков...")
                self.progress_signal.emit(5)
                
                # Создаем временный ZIP-файл
                temp_dir = Path(tempfile.gettempdir())
                temp_zip = temp_dir / f"client_upload_{int(time.time())}.zip"
                
                # Архивируем
                self._zip_dicom_dir(Path(self.dicom_dir), temp_zip)
                
                if self.is_cancelled:
                    raise RuntimeError("Операция отменена пользователем.")
                    
                self.step_signal.emit("Шаг 1 из 5: Отправка DICOM-архива на сервер...")
                self.progress_signal.emit(15)
                
                # Отправляем на сервер
                upload_url = f"{self.server_url.rstrip('/')}/api/jobs/upload"
                
                # Читаем zip-файл
                with open(temp_zip, 'rb') as f_zip:
                    files = {'file': ('dicom.zip', f_zip, 'application/zip')}
                    data = {
                        'client_name': self.client_name,
                        'options_json': json.dumps({
                            'preset_name': self.preset_name,
                            'precision_mode': self.precision_mode,
                            'selected_organs': self.selected_organs,
                            'merge_mode': self.merge_mode,
                            'use_gpu': self.use_gpu,
                            'remove_blobs': self.remove_blobs,
                            'smoothing_sigma': self.smoothing_sigma
                        })
                    }
                    
                    response = requests.post(upload_url, files=files, data=data, timeout=60)
                
                if response.status_code != 200:
                    raise RuntimeError(f"Сервер отклонил запрос: {response.text}")
                    
                res_data = response.json()
                job_id = res_data.get("job_id")
                self.job_id = job_id # сохраняем для отмены
                
                logger.info(f"Задача поставлена в очередь на сервере. ID: {job_id}")
                
                # Начинаем опрос статуса
                status_url = f"{self.server_url.rstrip('/')}/api/jobs/{job_id}/status"
                
                start_time = time.time()
                server_added_count = 0
                server_elapsed_seconds = 0.0
                self.last_log_index = 0
                while not self.is_cancelled:
                    time.sleep(2)
                    
                    try:
                        status_res = requests.get(status_url, timeout=10)
                        if status_res.status_code != 200:
                            raise RuntimeError(f"Ошибка получения статуса: {status_res.text}")
                            
                        status_data = status_res.json()
                        status = status_data.get("status")
                        progress = status_data.get("progress", 0)
                        step = status_data.get("current_step", "")
                        
                        # Обработка логов от TotalSegmentator и пайплайна
                        logs = status_data.get("logs", [])
                        if len(logs) > self.last_log_index:
                            for new_line in logs[self.last_log_index:]:
                                color = "#a0a0a2"
                                if "ERROR" in new_line or "Exception" in new_line or "failed" in new_line.lower():
                                    color = "#ff6b6b"
                                elif "WARNING" in new_line:
                                    color = "#f1c40f"
                                elif "шаг" in new_line.lower() or "---" in new_line:
                                    color = "#3498db"
                                elif "totalsegmentator" in new_line.lower():
                                    color = "#2ecc71"
                                self.log_signal.emit(new_line, color)
                            self.last_log_index = len(logs)
                        elapsed = status_data.get("elapsed_seconds", 0.0)
                        eta = status_data.get("eta_seconds", 0.0)
                        pos = status_data.get("queue_position")
                        is_server_paused = status_data.get("is_server_paused", False)
                        
                        # Отображение статуса
                        if status == "PENDING":
                            pause_prefix = " [СЕРВЕР НА ПАУЗЕ]" if is_server_paused else ""
                            self.step_signal.emit(f"В очереди (Позиция: {pos}){pause_prefix}...")
                            self.progress_signal.emit(15)
                        elif status == "PROCESSING":
                            self.step_signal.emit(step)
                            self.progress_signal.emit(progress)
                            self.eta_signal.emit(elapsed, eta)
                        elif status == "SUCCESS":
                            # Извлекаем количество добавленных ИИ структур из current_step
                            if "создано oar:" in step.lower():
                                try:
                                    server_added_count = int(step.lower().split("создано oar:")[-1].strip())
                                except ValueError:
                                    server_added_count = 0
                            server_elapsed_seconds = elapsed
                            break
                        elif status == "FAILED":
                            raise RuntimeError(status_data.get("error_message") or "Сбой вычислений на сервере.")
                        elif status == "CANCELLED":
                            raise RuntimeError("Операция отменена на сервере.")
                    except requests.exceptions.RequestException as re_err:
                        # Временный сбой связи в локальной сети: не паникуем, ждем
                        logger.warning(f"Микросбой локальной сети при опросе: {re_err}. Повторная попытка...")
                        self.step_signal.emit("Соединение с сервером потеряно... Попытка переподключения...")
                        time.sleep(3)
                
                if self.is_cancelled:
                    raise RuntimeError("Операция отменена пользователем.")
                    
                # Скачиваем результат
                self.step_signal.emit("Шаг 5 из 5: Скачивание результатов разметки...")
                self.progress_signal.emit(95)
                
                download_url = f"{self.server_url.rstrip('/')}/api/jobs/{job_id}/download"
                dl_res = requests.get(download_url, timeout=60)
                if dl_res.status_code != 200:
                    raise RuntimeError(f"Не удалось скачать файл разметки: {dl_res.text}")
                    
                # Сохраняем и распаковываем
                temp_result_zip = temp_dir / f"client_result_{job_id[:8]}.zip"
                with open(temp_result_zip, "wb") as f:
                    f.write(dl_res.content)
                    
                # Распаковываем воркспейс в выходной каталог
                out_path = Path(self.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(temp_result_zip, 'r') as zip_ref:
                    zip_ref.extractall(out_path)
                    
                # Чистим временный ZIP
                if temp_result_zip.exists():
                    temp_result_zip.unlink()
                    
                self.step_signal.emit("Готово! Контуры успешно импортированы.")
                self.progress_signal.emit(100)
                
                self.finished_signal.emit(True, f"Пайплайн успешно завершен! Добавлено структур: {server_added_count}. Общее время работы: {server_elapsed_seconds:.1f} сек.")
                
            except Exception as e:
                # В случае отмены посылаем DELETE на сервер
                if self.is_cancelled and job_id:
                    try:
                        logger.info(f"Отправка запроса на отмену задачи {job_id} на сервер...")
                        cancel_url = f"{self.server_url.rstrip('/')}/api/jobs/{job_id}/cancel"
                        requests.delete(cancel_url, timeout=5)
                    except Exception as ce:
                        logger.error(f"Не удалось отменить задачу на сервере: {ce}")
                
                self.finished_signal.emit(False, str(e))
                
            finally:
                if temp_zip and temp_zip.exists():
                    try:
                        temp_zip.unlink()
                    except Exception:
                        pass

    # Стилизация премиальной темной темы QSS
    DARK_QSS = """
    QWidget {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 13px;
    }

    QMenu {
        background-color: #242424;
        color: #ffffff;
        border: 1px solid #333333;
    }
    QMenu::item {
        background-color: transparent;
        padding: 6px 20px;
        color: #ffffff;
    }
    QMenu::item:selected {
        background-color: #007acc;
        color: #ffffff;
    }
    QMenu::item:disabled {
        color: #666666;
    }

    QToolTip {
        background-color: #2c2c2c;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 4px;
        padding: 4px;
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
        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0088ff, stop: 1 #0055cc);
        border: 1px solid #00aaff;
        color: #ffffff;
        font-weight: bold;
        padding: 6px 14px;
    }

    QPushButton#btnBrowse:hover {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #33a0ff, stop: 1 #0077ff);
        border: 1px solid #33ccff;
    }

    QPushButton#btnBrowse:pressed {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0044aa, stop: 1 #003388);
    }

    QPushButton#btnBrowse:disabled {
        background-color: #2b2b2b;
        border: 1px solid #3d3d3d;
        color: #888888;
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
        color: #ffffff;
    }

    QComboBox QAbstractItemView::item {
        color: #ffffff;
        background-color: #2d2d2d;
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

    QListWidget::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #666666;
        border-radius: 4px;
        background-color: #242424;
    }

    QListWidget::indicator:hover {
        border-color: #007acc;
        background-color: #2d2d2d;
    }

    QListWidget::indicator:checked {
        border-color: #007acc;
        background-color: #007acc;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE2LjJ6IiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+");
    }

    QListWidget::indicator:disabled {
        border-color: #444444;
        background-color: #1e1e1e;
    }

    QListWidget::indicator:checked:disabled {
        border-color: #444444;
        background-color: #444444;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE2LjJ6IiBmaWxsPSIjYWFhYWFhIi8+PC9zdmc+");
    }

    QRadioButton {
        spacing: 8px;
        color: #d0d0d0;
        margin-bottom: 6px;
        padding-top: 2px;
        padding-bottom: 2px;
    }

    QRadioButton::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #888888;
        border-radius: 10px;
        background-color: #242424;
    }

    QRadioButton::indicator:hover {
        border-color: #007acc;
        background-color: #2d2d2d;
    }

    QRadioButton::indicator:checked {
        border: 2px solid #007acc;
        border-radius: 10px;
        background-color: #007acc;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9JzAgMCAyNCAyNCcgd2lkdGg9JzI0JyBoZWlnaHQ9JzI0Jz48Y2lyY2xlIGN4PScxMicgY3k9JzEyJyByPSc2JyBmaWxsPScjZmZmZmZmJy8+PC9zdmc+");
    }

    QRadioButton::indicator:disabled {
        border-color: #444444;
        background-color: #1e1e1e;
    }

    QRadioButton::indicator:checked:disabled {
        border: 2px solid #444444;
        border-radius: 10px;
        background-color: #444444;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9JzAgMCAyNCAyNCcgd2lkdGg9JzI0JyBoZWlnaHQ9JzI0Jz48Y2lyY2xlIGN4PScxMicgY3k9JzEyJyByPSc2JyBmaWxsPScjODg4ODg4Jy8+PC9zdmc+");
    }

    QRadioButton::disabled {
        color: #666666;
    }

    QCheckBox {
        spacing: 8px;
        color: #d0d0d0;
    }

    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #666666;
        border-radius: 4px;
        background-color: #242424;
    }

    QCheckBox::indicator:hover {
        border-color: #007acc;
        background-color: #2d2d2d;
    }

    QCheckBox::indicator:checked {
        border-color: #007acc;
        background-color: #007acc;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE2LjJ6IiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+");
    }

    QCheckBox::indicator:disabled {
        border-color: #444444;
        background-color: #1e1e1e;
    }

    QCheckBox::indicator:checked:disabled {
        border-color: #444444;
        background-color: #444444;
        image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE2LjJ6IiBmaWxsPSIjYWFhYWFhIi8+PC9zdmc+");
    }

    QCheckBox::disabled {
        color: #666666;
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

    QTabWidget::pane {
        border: 1px solid #333333;
        border-radius: 6px;
        background: #242424;
        padding: 10px;
    }

    QTabBar::tab {
        background: #1e1e1e;
        border: 1px solid #333333;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: #a0a0a0;
    }
    QTabBar::tab:selected {
        background: #242424;
        border-bottom-color: #242424;
        color: #ffffff;
    }

    QTabBar::tab:hover {
        background: #2b2b2b;
    }

    QGroupBox {
        border: 1px solid #333333;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 15px;
        font-weight: bold;
        color: #ffffff;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
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

    class LicensesDialog(QDialog):
        """Диалоговое окно управления единой лицензией суб-моделей ИИ."""
        def __init__(self, parent=None, engine=None):
            super().__init__(parent)
            self.engine = engine
            self.setWindowTitle("🔑 Лицензирование суб-моделей TotalSegmentator")
            self.setMinimumSize(480, 260)
            self.init_ui()
            
        def init_ui(self):
            self.setStyleSheet(DARK_QSS)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Заголовок
            title = QLabel("🔑 Лицензия для суб-моделей")
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            layout.addWidget(title)
            

            
            # Статус текущей лицензии в красивой плашке
            self.status_frame = QFrame()
            self.status_frame.setStyleSheet("""
                QFrame {
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            status_layout = QHBoxLayout(self.status_frame)
            status_layout.setContentsMargins(10, 5, 10, 5)
            
            self.lbl_status = QLabel("Статус: ❌ Отсутствует")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #ff6b6b;")
            
            self.lbl_key = QLabel("Ключ: —")
            self.lbl_key.setStyleSheet("color: #a0a0a0; font-family: monospace; font-size: 12px;")
            
            status_layout.addWidget(self.lbl_status)
            status_layout.addStretch()
            status_layout.addWidget(self.lbl_key)
            layout.addWidget(self.status_frame)
            
            # Поле ввода ключа
            self.edit_key = QLineEdit()
            self.edit_key.setPlaceholderText("Введите новый лицензионный ключ...")
            self.edit_key.setMinimumHeight(30)
            layout.addWidget(self.edit_key)
            
            # Кнопки управления
            btn_layout = QHBoxLayout()
            self.btn_save = QPushButton("💾 Сохранить и проверить")
            self.btn_save.setObjectName("btnBrowse")  # синяя кнопка
            self.btn_save.setMinimumHeight(32)
            self.btn_save.clicked.connect(self.save_license)
            
            self.btn_delete = QPushButton("🗑️ Удалить")
            self.btn_delete.setObjectName("btnAction")
            self.btn_delete.setMinimumHeight(32)
            self.btn_delete.clicked.connect(self.delete_license)
            
            btn_layout.addWidget(self.btn_save, 2)
            btn_layout.addWidget(self.btn_delete, 1)
            layout.addLayout(btn_layout)
            
            # Кнопка закрытия
            self.btn_close = QPushButton("Закрыть")
            self.btn_close.setMinimumHeight(30)
            self.btn_close.clicked.connect(self.accept)
            layout.addWidget(self.btn_close)
            
            self.update_status_display()
            
        def update_status_display(self):
            key = getattr(self.engine, "licenses", "").strip()
            if key:
                self.lbl_status.setText("Статус: ✅ Активна")
                self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #00ffd0;")
                
                # Маскируем ключ для приватности, показывая только начало и конец (например: aca_4HM5...ODB2)
                masked_key = key
                if len(key) >= 12:
                    masked_key = f"{key[:8]}...{key[-4:]}"
                self.lbl_key.setText(f"Ключ: {masked_key}")
            else:
                self.lbl_status.setText("Статус: ❌ Отсутствует")
                self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #ff6b6b;")
                self.lbl_key.setText("Ключ: —")
                
        def save_license(self):
            key = self.edit_key.text().strip()
            if not key:
                QMessageBox.warning(self, "Предупреждение", "Лицензионный ключ не может быть пустым.")
                return
                
            # 1. Локальная проверка длины
            if len(key) != 18:
                QMessageBox.critical(self, "Ошибка валидации", "Лицензионный ключ должен содержать ровно 18 символов.")
                return
                
            # 2. Онлайн-валидация через API TotalSegmentator
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                from totalsegmentator.config import is_valid_license
                valid = is_valid_license(key)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                reply = QMessageBox.question(
                    self,
                    "Ошибка сети",
                    f"Не удалось связаться с сервером TotalSegmentator для проверки лицензии ({e}).\n\n"
                    "Формат ключа верный. Хотите сохранить его локально без онлайн-проверки?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.engine.licenses = key
                    self.engine.save_presets_config()
                    self._write_license_to_totalseg_config(key)
                    self.edit_key.clear()
                    self.update_status_display()
                    QMessageBox.information(self, "Успех", "Лицензия сохранена локально.")
                return
                
            QApplication.restoreOverrideCursor()
            
            if not valid:
                QMessageBox.critical(
                    self,
                    "Недействительная лицензия ❌",
                    "Введенный лицензионный ключ недействителен.\n"
                    "Проверка на сервере TotalSegmentator отклонена. Пожалуйста, убедитесь в правильности ключа."
                )
                return
                
            # Сохранение валидной лицензии
            self.engine.licenses = key
            self.engine.save_presets_config()
            self._write_license_to_totalseg_config(key)
            self.edit_key.clear()
            self.update_status_display()
            QMessageBox.information(self, "Успех", "Лицензия для суб-моделей успешно активирована и сохранена! ✅")
            
        def delete_license(self):
            key = getattr(self.engine, "licenses", "").strip()
            if not key:
                QMessageBox.information(self, "Инфо", "Лицензионный ключ отсутствует.")
                return
                
            reply = QMessageBox.question(
                self,
                "Подтверждение удаления",
                "Вы действительно хотите удалить лицензионный ключ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.engine.licenses = ""
                self.engine.save_presets_config()
                self._write_license_to_totalseg_config("")
                self.update_status_display()
                QMessageBox.information(self, "Успех", "Лицензионный ключ успешно удален.")

        def _write_license_to_totalseg_config(self, key: str):
            """Записывает лицензионный ключ напрямую в config.json TotalSegmentator."""
            try:
                from totalsegmentator.libs import get_totalseg_dir
                import json
                totalseg_dir = get_totalseg_dir()
                totalseg_dir.mkdir(parents=True, exist_ok=True)
                totalseg_config_file = totalseg_dir / "config.json"
                
                cfg = {}
                if totalseg_config_file.exists():
                    try:
                        with open(totalseg_config_file, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                    except Exception:
                        pass
                
                cfg["license_number"] = key.strip()
                with open(totalseg_config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=4)
                logger.info(f"Лицензия успешно записана в файл конфигурации TotalSegmentator: {totalseg_config_file}")
            except Exception as e:
                logger.error(f"Не удалось записать лицензию в файл конфигурации TotalSegmentator: {e}")

    # =========================================================================
    #                    УПРАВЛЕНИЕ МОДЕЛЯМИ ИИ (TotalSegmentator)
    # =========================================================================

    MODELS_LIST = [
        {
            "id": 291,
            "task": "total_part1",
            "folder": "Dataset291_TotalSegmentator_part1_organs_1559subj",
            "name": "Базовая модель (Часть 1: Внутренние органы)",
            "licensed": False
        },
        {
            "id": 292,
            "task": "total_part2",
            "folder": "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
            "name": "Базовая модель (Часть 2: Позвоночник)",
            "licensed": False
        },
        {
            "id": 293,
            "task": "total_part3",
            "folder": "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
            "name": "Базовая модель (Часть 3: Сердце и сосуды)",
            "licensed": False
        },
        {
            "id": 294,
            "task": "total_part4",
            "folder": "Dataset294_TotalSegmentator_part4_muscles_1559subj",
            "name": "Базовая модель (Часть 4: Мышцы тела)",
            "licensed": False
        },
        {
            "id": 295,
            "task": "total_part5",
            "folder": "Dataset295_TotalSegmentator_part5_ribs_1559subj",
            "name": "Базовая модель (Часть 5: Ребра)",
            "licensed": False
        },
        {
            "id": 775,
            "task": "head_glands_cavities",
            "folder": "Dataset775_head_glands_cavities_492subj",
            "name": "Голова (Слюнные железы и полости)",
            "licensed": False
        },
        {
            "id": 409,
            "task": "brain_structures",
            "folder": "Dataset409_neuro_550subj",
            "name": "Отделы головного мозга (Brain Structures)",
            "licensed": True
        },
        {
            "id": 304,
            "task": "appendicular_bones",
            "folder": "Dataset304_appendicular_bones_ext_1559subj",
            "name": "Кости конечностей (Appendicular Bones)",
            "licensed": True
        },
        {
            "id": 301,
            "task": "heartchambers_highres",
            "folder": "Dataset301_heart_highres_1559subj",
            "name": "Камеры сердца высокого разрешения",
            "licensed": True
        }
    ]

    class DownloadWorker(QThread):
        """Фоновый поток для неблокирующего скачивания весов модели ИИ с прогресс-баром."""
        progress = pyqtSignal(int)
        status_msg = pyqtSignal(str)
        finished = pyqtSignal(bool, str)

        def __init__(self, task_id):
            super().__init__()
            self.task_id = task_id

        def run(self):
            try:
                import os
                import zipfile
                import requests
                from totalsegmentator.config import get_weights_dir
                from totalsegmentator.libs import get_version
                
                config_dir = get_weights_dir()
                config_dir.mkdir(exist_ok=True, parents=True)
                tempfile = config_dir / "tmp_download_file.zip"
                
                commercial_tasks = {
                    409: "brain_structures",
                    304: "appendicular_bones",
                    301: "heartchambers_highres"
                }
                
                public_urls = {
                    291: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
                    292: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
                    293: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
                    294: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
                    295: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
                    775: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.3.0-weights/Dataset775_head_glands_cavities_492subj.zip"
                }
                
                if self.task_id in commercial_tasks:
                    from totalsegmentator.libs import get_totalseg_dir
                    import json
                    totalseg_dir = get_totalseg_dir()
                    totalseg_config_file = totalseg_dir / "config.json"
                    
                    if totalseg_config_file.exists():
                        with open(totalseg_config_file) as f:
                            cfg = json.load(f)
                        license_number = cfg.get("license_number", "")
                    else:
                        raise ValueError(f"Не найден файл конфигурации лицензии: {totalseg_config_file}")
                    
                    if not license_number:
                        raise ValueError("Отсутствует активный лицензионный ключ.")
                    
                    url = "https://backend.totalsegmentator.com:443/download_weights"
                    payload = {
                        "license_number": license_number,
                        "task": commercial_tasks[self.task_id],
                        "version": get_version()
                    }
                    
                    r = requests.post(url, json=payload, timeout=300, stream=True)
                else:
                    if self.task_id not in public_urls:
                        raise ValueError(f"Неизвестный ID задачи {self.task_id} для скачивания.")
                    url = public_urls[self.task_id]
                    r = requests.get(url, stream=True, timeout=300)
                
                r.raise_for_status()
                
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(tempfile, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = int((downloaded / total_size) * 100)
                                self.progress.emit(percent)
                
                self.status_msg.emit("Загрузка завершена! Распаковка файлов модели...\nПожалуйста, подождите, это займет несколько секунд.")
                
                with zipfile.ZipFile(tempfile, 'r') as zip_f:
                    zip_f.extractall(config_dir)
                
                if tempfile.exists():
                    tempfile.unlink()
                    
                self.finished.emit(True, "")
            except Exception as e:
                try:
                    if 'tempfile' in locals() and tempfile.exists():
                        tempfile.unlink()
                except Exception:
                    pass
                self.finished.emit(False, str(e))

    class ModelsDialog(QDialog):
        """Диалоговое окно управления моделями ИИ TotalSegmentator."""
        def __init__(self, parent=None, engine=None):
            super().__init__(parent)
            self.engine = engine
            self.setWindowTitle("📦 Управление моделями ИИ TotalSegmentator")
            self.setMinimumSize(720, 680)
            self.resize(720, 680)
            self.init_ui()

        def init_ui(self):
            self.setStyleSheet(DARK_QSS)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)

            # Заголовок
            title = QLabel("📦 Доступные модели ИИ")
            title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            layout.addWidget(title)

            # Описание
            descr = QLabel(
                "Здесь отображается список всех доступных для скачивания суб-моделей TotalSegmentator.\n"
                "Если модель не скачана, перед началом сегментации потребуется время на её загрузку."
            )
            descr.setWordWrap(True)
            descr.setStyleSheet("color: #a0a0a0; font-size: 12px; line-height: 1.4;")
            layout.addWidget(descr)

            # Таблица моделей
            self.table = QTableWidget()
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["Модель (Распознаваемая область)", "Статус", "Действие"])
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(1, 110)
            self.table.setColumnWidth(2, 180)
            
            # Настройка внешнего вида таблицы
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            
            # Настройка высоты строк и скрытие номеров строк для чистоты интерфейса
            self.table.verticalHeader().setVisible(False)
            self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            self.table.verticalHeader().setDefaultSectionSize(50)
            
            self.table.setStyleSheet("""
                QTableWidget {
                    background-color: #1e1e1e;
                    gridline-color: #2d2d2d;
                    border: 1px solid #2d2d2d;
                    border-radius: 6px;
                }
                QHeaderView::section {
                    background-color: #242424;
                    color: #ffffff;
                    padding: 8px;
                    border: 1px solid #2d2d2d;
                    font-weight: bold;
                    font-size: 12px;
                }
                QTableWidget::item {
                    padding: 8px;
                    color: #e0e0e0;
                    font-size: 12px;
                }
            """)
            layout.addWidget(self.table)

            # Кнопка закрытия
            self.btn_close = QPushButton("Закрыть")
            self.btn_close.setMinimumHeight(32)
            self.btn_close.clicked.connect(self.accept)
            layout.addWidget(self.btn_close)

            self.populate_table()

        def populate_table(self):
            from totalsegmentator.config import get_weights_dir
            weights_dir = get_weights_dir()
            
            self.table.setRowCount(len(MODELS_LIST))
            
            for row, model in enumerate(MODELS_LIST):
                # 1. Название и описание
                name_text = model["name"]
                if model["licensed"]:
                    name_text += " 🔑 (Нужна лицензия)"
                
                name_item = QTableWidgetItem(name_text)
                name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, 0, name_item)
                
                # 2. Статус
                folder_path = weights_dir / model["folder"]
                is_downloaded = folder_path.exists()
                
                status_item = QTableWidgetItem()
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                if is_downloaded:
                    status_item.setText("✅ Доступна")
                    status_item.setForeground(QBrush(QColor("#00ffd0"))) # Бирюзовый
                else:
                    status_item.setText("❌ Не скачана")
                    status_item.setForeground(QBrush(QColor("#ff6b6b"))) # Кораллово-красный
                self.table.setItem(row, 1, status_item)
                
                # 3. Кнопка действия (Скачать / Удалить)
                btn = QPushButton()
                if is_downloaded:
                    btn.setText("🗑️ Удалить")
                    btn.setObjectName("btnAction")
                    btn.clicked.connect(lambda checked, m=model: self.delete_model(m))
                else:
                    btn.setText("📥 Скачать")
                    btn.setObjectName("btnBrowse")
                    btn.clicked.connect(lambda checked, m=model: self.download_model(m))
                
                # Центрируем кнопку в ячейке, чтобы избежать растяжения и обрезания краев
                cell_widget = QWidget()
                cell_layout = QHBoxLayout(cell_widget)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell_layout.addWidget(btn)
                self.table.setCellWidget(row, 2, cell_widget)
                
                # Задаем фиксированную высоту строки во избежание сжатия
                self.table.setRowHeight(row, 50)

        def delete_model(self, model):
            from totalsegmentator.config import get_weights_dir
            from totalsegmentator.libs import robust_rmtree
            
            reply = QMessageBox.question(
                self,
                "Подтверждение удаления весов ИИ",
                f"Вы действительно хотите удалить локальные веса для модели:\n\n\"{model['name']}\"?\n\n"
                "При следующем использовании данной модели её веса будут скачиваться заново.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                folder_path = get_weights_dir() / model["folder"]
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    if folder_path.exists():
                        robust_rmtree(folder_path)
                    QApplication.restoreOverrideCursor()
                    QMessageBox.information(self, "Успех", f"Веса модели \"{model['name']}\" успешно удалены.")
                    self.populate_table()
                except Exception as e:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.critical(self, "Ошибка удаления", f"Не удалось удалить папку весов:\n{e}")

        def download_model(self, model):
            import json
            # Проверка лицензии для коммерческих моделей
            if model["licensed"]:
                license_key = self.engine.licenses.strip() if hasattr(self.engine, "licenses") and isinstance(self.engine.licenses, str) else ""
                if not license_key:
                    QMessageBox.critical(
                        self,
                        "Требуется лицензия 🔑",
                        f"Модель \"{model['name']}\" является коммерческой суб-моделью.\n\n"
                        "Для её скачивания необходимо активировать лицензионный ключ.\n\n"
                        "Пожалуйста, перейдите во вкладку 'Настройки' -> 'Управление лицензиями' и введите ключ."
                    )
                    return
                
                # Перед скачиванием пишем лицензию в ~/.totalsegmentator/config.json
                try:
                    from totalsegmentator.libs import get_totalseg_dir
                    totalseg_dir = get_totalseg_dir()
                    totalseg_dir.mkdir(exist_ok=True, parents=True)
                    config_file = totalseg_dir / "config.json"
                    
                    # Пытаемся считать существующий конфиг
                    config_data = {}
                    if config_file.exists():
                        try:
                            with open(config_file, "r") as f:
                                config_data = json.load(f)
                        except Exception:
                            pass
                    
                    config_data["license_number"] = license_key
                    with open(config_file, "w") as f:
                        json.dump(config_data, f)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Внимание",
                        f"Не удалось автоматически записать лицензионный ключ в конфигурационный файл TotalSegmentator ({e}).\n"
                        "Скачивание может завершиться ошибкой."
                    )
            
            # Запускаем фоновое скачивание
            progress = QProgressDialog(
                f"Идет скачивание весов модели \"{model['name']}\"...\n"
                "Пожалуйста, подождите. Это может занять несколько минут в зависимости от скорости интернета.",
                None, 0, 100, self
            )
            progress.setWindowTitle("📥 Загрузка весов модели ИИ")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            # Отключаем кнопку "Отмена", так как синхронная функция не поддерживает отмену во время скачивания
            progress.setCancelButton(None)
            
            # Запускаем поток
            self.dl_worker = DownloadWorker(model["id"])
            
            # Подключаем сигналы для обновления прогресса и статуса
            self.dl_worker.progress.connect(progress.setValue)
            self.dl_worker.status_msg.connect(progress.setLabelText)
            
            def on_finished(success, err_msg):
                progress.close()
                if success:
                    QMessageBox.information(
                        self,
                        "Успех ✅",
                        f"Веса модели \"{model['name']}\" успешно загружены и готовы к работе!"
                    )
                    self.populate_table()
                else:
                    QMessageBox.critical(
                        self,
                        "Ошибка скачивания ❌",
                        f"Произошла ошибка при скачивании модели \"{model['name']}\":\n\n{err_msg}"
                    )
            
            self.dl_worker.finished.connect(on_finished)
            self.dl_worker.start()
            progress.exec()

    class NonScrollComboBox(QComboBox):
        """Выпадающий список QComboBox, который игнорирует событие прокрутки колесика мыши во избежание случайных изменений."""
        def __init__(self, parent=None):
            super().__init__(parent)
            from PyQt6.QtWidgets import QListView
            self.setView(QListView())

        def wheelEvent(self, event):
            event.ignore()

    class QueueTableWidget(QTableWidget):
        """Таблица с поддержкой Drag & Drop для управления очередью задач."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDragDropOverwriteMode(False)
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.parent_app = None

        def dragEnterEvent(self, event):
            row = self.currentRow()
            if row < 0:
                event.ignore()
                return
            status_item = self.item(row, 4)
            if status_item and status_item.text() == "PROCESSING":
                event.ignore()
                return
            event.acceptProposedAction()

        def dragMoveEvent(self, event):
            event.acceptProposedAction()

        def dropEvent(self, event):
            source_row = self.currentRow()
            if source_row < 0:
                event.ignore()
                return

            target_index = self.indexAt(event.position().toPoint())
            target_row = target_index.row()
            if target_row < 0:
                target_row = self.rowCount() - 1

            has_processing = False
            first_row_status = self.item(0, 4)
            if first_row_status and first_row_status.text() == "PROCESSING":
                has_processing = True

            if has_processing:
                if target_row == 0:
                    target_row = 1
                if source_row == 0:
                    event.ignore()
                    return

            if source_row == target_row:
                event.ignore()
                return

            event.accept()
            
            # Собираем ID задач в новом порядке
            new_job_ids = []
            for r in range(self.rowCount()):
                if r == source_row:
                    continue
                job_id = self.item(r, 0).data(Qt.ItemDataRole.UserRole)
                if job_id:
                    new_job_ids.append(job_id)
            
            moved_job_id = self.item(source_row, 0).data(Qt.ItemDataRole.UserRole)
            if moved_job_id:
                new_job_ids.insert(target_row, moved_job_id)
                
            if self.parent_app:
                self.parent_app.send_new_queue_order(new_job_ids)

    class MainWindow(QMainWindow):
        """Главное окно графического интерфейса приложения."""
        def __init__(self):
            super().__init__()
            self.setWindowTitle("AI Contour — Автооконтурирование органов риска (сервер + лок. клиент")
            self.setMinimumSize(960, 760)
            self.showMaximized()
            self.existing_rtstruct_path = None
            self.is_updating_presets = False
            self.collapsed_groups = {"Остальное": True}
            self.worker = None
            self.settings = QSettings("AIContourCorp", "AIContour")

            # Инициализация вычислительного движка
            self.engine = ContourEngine()
            self.stats_manager = StatisticsManager()

            # Настройка перенаправления логов в реальном времени
            self.log_signaler = LogSignaler()
            self.log_signaler.log_signal.connect(self.append_log)
            self.log_handler = QTextEditLogHandler(self.log_signaler)
            logging.getLogger().addHandler(self.log_handler)

            self.pause_signaler = PauseSignaler()
            self.pause_signaler.pause_signal.connect(self.update_gui_after_pause)

            self.server_status_signaler = ServerStatusSignaler()
            self.server_status_signaler.status_signal.connect(self.handle_server_status_update)

            # Таймер активности (спиннер + пульсация цвета)
            self.activity_timer = QTimer(self)
            self.activity_timer.setInterval(120)
            self.activity_timer.timeout.connect(self.update_activity_animation)
            self.spinner_index = 0
            self.pulse_tick = 0
            self.current_step_base_text = "Ожидание запуска..."
            self.SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

            # Таймер фонового сканирования
            self.scan_timer = QTimer(self)
            self.scan_timer.setInterval(15000)
            self.scan_timer.timeout.connect(self.start_background_scan)

            # Инициализируем переменные состояния сервера
            self.server_is_paused = False
            self.is_toggling_pause = False
            self.server_process = None
            self.start_server_process()

            self.init_ui()
            
            # Выведем лог стартапа, если есть буферизованные записи
            if hasattr(self, '_startup_log_buffer'):
                for msg, col in self._startup_log_buffer:
                    self.append_log(msg, col)
                del self._startup_log_buffer

            self.load_settings()

            # Таймер для обновления состояния сервера и очереди задач
            self.server_ui_timer = QTimer(self)
            self.server_ui_timer.setInterval(1000)
            self.server_ui_timer.timeout.connect(self.update_server_ui)
            self.server_ui_timer.start()

        def start_server_process(self):
            import subprocess
            import sys
            logging.info("Очистка порта 8000 от старых процессов...")
            if os.name == 'nt':
                try:
                    netstat_out = subprocess.check_output("netstat -ano", shell=True, text=True)
                    pids_to_kill = set()
                    current_pid = str(os.getpid())
                    for line in netstat_out.strip().split("\n"):
                        if ":8000" in line and "LISTENING" in line:
                            parts = [p.strip() for p in line.split(" ") if p.strip()]
                            if len(parts) >= 5:
                                pid = parts[-1]
                                if pid.isdigit() and pid != "0" and pid != current_pid:
                                    pids_to_kill.add(pid)
                    for pid in pids_to_kill:
                        logging.info(f"Жесткое завершение старого процесса прослушивания PID {pid}...")
                        subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as ex:
                    logging.debug(f"Не удалось очистить порт 8000: {ex}")
            
            logging.info("Запуск фонового процесса FastAPI бэкенда...")
            try:
                creation_flags = 0
                if os.name == 'nt':
                    creation_flags = subprocess.CREATE_NO_WINDOW
                
                app_dir = os.path.dirname(os.path.abspath(__file__))
                logs_dir = os.path.join(app_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                stdout_log_path = os.path.join(logs_dir, "server_stdout.log")
                
                # Ротация файла server_stdout.log перед запуском (до 2 файлов по 5 МБ)
                try:
                    if os.path.exists(stdout_log_path):
                        if os.path.getsize(stdout_log_path) > 5 * 1024 * 1024:
                            backup_log_path = stdout_log_path + ".1"
                            if os.path.exists(backup_log_path):
                                os.remove(backup_log_path)
                            os.rename(stdout_log_path, backup_log_path)
                            logging.info("Выполнена ротация файла server_stdout.log")
                except Exception as re:
                    logging.debug(f"Не удалось выполнить ротацию server_stdout.log: {re}")
                
                # Открываем файл логов в режиме перезаписи/дозаписи
                self.server_stdout_file = open(stdout_log_path, "a", encoding="utf-8")
                self.server_stdout_file.write(f"\n--- ЗАПУСК СЕРВЕРА: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                self.server_stdout_file.flush()
                
                self.server_process = subprocess.Popen(
                    [sys.executable, "server/main.py"],
                    stdout=self.server_stdout_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    cwd=app_dir
                )
                logging.info(f"Процесс FastAPI сервера запущен. Логи бэкенда пишутся в: {stdout_log_path}")
            except Exception as e:
                logging.error(f"Не удалось запустить процесс сервера: {e}")
                QMessageBox.critical(self, "Ошибка сервера", f"Не удалось запустить серверный процесс: {e}")

        def init_ui(self):
            # Установка премиальной иконки приложения
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))

            self.setStyleSheet(DARK_QSS)

            # Главный виджет
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QVBoxLayout(main_widget)
            main_layout.setContentsMargins(15, 15, 15, 15)
            main_layout.setSpacing(10)

            # Сплиттер
            self.splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(self.splitter, 1)

            # --- ЛЕВАЯ КОЛОНКА (Вкладки настроек) ---
            self.left_card = QFrame()
            self.left_card.setObjectName("card")
            self.left_card.setMinimumWidth(490)
            left_layout = QVBoxLayout(self.left_card)
            left_layout.setContentsMargins(5, 5, 5, 5)

            # Панель управления сервером (кнопка паузы + адрес API)
            server_header_widget = QWidget()
            server_header_layout = QVBoxLayout(server_header_widget)
            server_header_layout.setContentsMargins(10, 5, 10, 5)
            server_header_layout.setSpacing(4)
            
            self.btn_pause_toggle = QPushButton("СЕРВЕР АКТИВЕН 🟢")
            self.btn_pause_toggle.setObjectName("btnPauseActive")
            self.btn_pause_toggle.setStyleSheet("""
                QPushButton#btnPauseActive {
                    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #27ae60, stop: 1 #1e8449);
                    border: 1px solid #2ecc71;
                    color: #ffffff;
                    padding: 8px 18px;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            self.btn_pause_toggle.clicked.connect(self.toggle_pause)
            
            self.lbl_server_address = QLabel("Инициализация сервера...")
            self.lbl_server_address.setStyleSheet("font-size: 12px; color: #3498db; font-weight: 500;")
            
            server_header_layout.addWidget(self.btn_pause_toggle)
            server_header_layout.addWidget(self.lbl_server_address)
            left_layout.addWidget(server_header_widget)

            self.tab_widget = QTabWidget()
            self.tab_widget.currentChanged.connect(self.on_tab_changed)
            left_layout.addWidget(self.tab_widget)

            # ------------------------------------------------------------------
            # ВКЛАДКА 1: Выбор органов и снимков
            # ------------------------------------------------------------------
            tab1_widget = QWidget()
            tab1_layout = QVBoxLayout(tab1_widget)
            tab1_layout.setSpacing(10)

            # Выбор пресета
            preset_label = QLabel("Выбор пресета органов (OAR):")
            preset_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.preset_combo = NonScrollComboBox()
            # activated срабатывает при каждом выборе из списка, даже если значение не изменилось
            self.preset_combo.activated.connect(self.on_preset_changed)
            tab1_layout.addWidget(preset_label)
            tab1_layout.addWidget(self.preset_combo)

            # Кнопки быстрого выделения
            selection_layout = QHBoxLayout()
            self.btn_select_all = QPushButton("Выбрать все")
            self.btn_select_all.setObjectName("btnAction")
            self.btn_select_all.clicked.connect(self.select_all_organs)
            
            self.btn_deselect_all = QPushButton("Снять все")
            self.btn_deselect_all.setObjectName("btnAction")
            self.btn_deselect_all.clicked.connect(self.deselect_all_organs)
            
            selection_layout.addWidget(self.btn_select_all)
            selection_layout.addWidget(self.btn_deselect_all)
            tab1_layout.addLayout(selection_layout)

            # Список OAR с чек-боксами
            self.organs_header = QLabel("Органы для автооконтурирования: 0 из 0")
            self.organs_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.organs_list = QListWidget()
            self.organs_list.itemChanged.connect(self.on_organ_item_changed)
            self.organs_list.itemClicked.connect(self.on_organ_item_clicked)
            self.imported_items = []

            tab1_layout.addWidget(self.organs_header)
            tab1_layout.addWidget(self.organs_list)
            
            # Двойной клик по элементу списка для выбора цвета
            self.organs_list.itemDoubleClicked.connect(self.pick_organ_color)
            
            self.tab_widget.addTab(tab1_widget, "🎯 Контуры")

            # ------------------------------------------------------------------
            # ВКЛАДКА 2: Параметры ИИ и Цвета
            # ------------------------------------------------------------------
            tab2_scroll = QScrollArea()
            tab2_scroll.setWidgetResizable(True)
            tab2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tab2_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

            tab2_widget = QWidget()
            tab2_widget.setObjectName("tab2_widget")
            tab2_widget.setStyleSheet("background-color: transparent;")
            tab2_layout = QVBoxLayout(tab2_widget)
            tab2_layout.setContentsMargins(10, 10, 10, 10)
            tab2_layout.setSpacing(12)

            # Группа 0: Выбор КТ DICOM (перенесено с главной)
            input_group = QGroupBox("Папка с КТ-снимками DICOM")
            input_group_layout = QVBoxLayout(input_group)
            
            self.input_edit = QLineEdit()
            self.input_edit.setPlaceholderText("Выберите папку с DICOM файлами...")
            self.input_edit.textChanged.connect(self.check_for_rtstruct)
            self.btn_input = QPushButton("📂 Источник")
            self.btn_input.setObjectName("btnBrowse")
            self.btn_input.clicked.connect(self.select_input_dir)

            input_box = QHBoxLayout()
            input_box.addWidget(self.input_edit)
            input_box.addWidget(self.btn_input)
            input_group_layout.addLayout(input_box)
            tab2_layout.addWidget(input_group)
            
            # Группа: Действия с файлами структур (перенесено из Tab 1)
            merge_group = QGroupBox("Действия с файлами структур")
            merge_group_layout = QVBoxLayout(merge_group)
            merge_group_layout.setSpacing(10)
            
            self.status_rtstruct_label = QLabel("RTSTRUCT: путь не выбран")
            self.status_rtstruct_label.setStyleSheet("color: #888888;")
            self.status_rtstruct_label.setWordWrap(True)
            merge_group_layout.addWidget(self.status_rtstruct_label)
            
            self.merge_btn_group = QButtonGroup(self)
            self.radio_merge_new = QRadioButton("Создать новый файл RTSTRUCT")
            self.radio_merge_merge = QRadioButton("Дополнить существующий файл")
            
            self.radio_merge_new.setChecked(True)
            self.merge_btn_group.addButton(self.radio_merge_new, 1)
            self.merge_btn_group.addButton(self.radio_merge_merge, 2)
            
            merge_group_layout.addWidget(self.radio_merge_new)
            merge_group_layout.addWidget(self.radio_merge_merge)
            
            # Выпадающий список выбора целевого файла для дополнения
            self.merge_rtstruct_combo = NonScrollComboBox()
            self.merge_rtstruct_combo.setEnabled(False)
            self.merge_rtstruct_combo.setStyleSheet("margin-left: 20px; padding: 4px;")
            merge_group_layout.addWidget(self.merge_rtstruct_combo)
            
            self.radio_merge_new.toggled.connect(self.update_merge_combo_state)
            self.radio_merge_merge.toggled.connect(self.update_merge_combo_state)
            self.merge_rtstruct_combo.currentIndexChanged.connect(self.on_merge_rtstruct_changed)
            
            tab2_layout.addWidget(merge_group)
            
            gpu_available = self.engine.is_gpu_available()

            # Группа 1: Вычислительное устройство
            device_group = QGroupBox("Вычислительное устройство")
            device_group_layout = QVBoxLayout(device_group)
            device_group_layout.setSpacing(10)
            self.radio_cpu = QRadioButton("CPU (Центральный процессор)")
            self.radio_gpu = QRadioButton("GPU CUDA (Рекомендуется)")
            
            # На клиенте обе опции (GPU/CPU) всегда доступны для выбора, так как вычисления идут на сервере
            self.radio_gpu.setChecked(True)
            self.radio_gpu.setEnabled(True)
                
            device_group_layout.addWidget(self.radio_gpu)
            device_group_layout.addWidget(self.radio_cpu)
            tab2_layout.addWidget(device_group)

            # Группа 2: Режимы точности TotalSegmentator
            precision_group = QGroupBox("Точность и разрешение ИИ")
            precision_group_layout = QVBoxLayout(precision_group)
            
            self.precision_combo = NonScrollComboBox()
            self.precision_combo.addItems([
                "Стандартная (1.5 мм)",
                "Быстрая (3.0 мм)",
                "Ультра-быстрая (Body контур тела)"
            ])
            self.precision_combo.setToolTip(
                "Стандартная: высокое разрешение контуров (1.5 мм)\n"
                "Быстрая: сниженное разрешение (3 мм), скорость выше в 3-4 раза\n"
                "Ультра-быстрая: только для разметки внешнего контура тела целиком"
            )
            precision_group_layout.addWidget(self.precision_combo)
            tab2_layout.addWidget(precision_group)

            # Группа 3: 3D Постобработка масок
            post_group = QGroupBox("Постобработка 3D масок")
            post_group_layout = QVBoxLayout(post_group)
            
            self.clean_blobs_check = QCheckBox("Удалять мелкие артефакты (Blobs)")
            self.clean_blobs_check.setToolTip(
                "Удаляет изолированный мелкий шум нейросети на КТ-срезах,\n"
                "оставляя только основной объем органа."
            )
            self.clean_blobs_check.setChecked(True)
            
            self.smoothing_check = QCheckBox("Сглаживание контуров (Smoothing)")
            self.smoothing_check.setToolTip(
                "Применяет Гауссову фильтрацию к 3D-маске, убирая «ступенчатость» срезов."
            )
            self.smoothing_check.stateChanged.connect(self.on_smoothing_check_changed)
            
            smoothing_param_layout = QHBoxLayout()
            smoothing_param_label = QLabel("Уровень сглаживания:")
            self.smoothing_combo = NonScrollComboBox()
            self.smoothing_combo.addItems([
                "Легкое (sigma = 0.5)",
                "Стандартное (sigma = 1.0)",
                "Сильное (sigma = 1.5)",
                "Максимальное (sigma = 2.0)"
            ])
            self.smoothing_combo.setCurrentIndex(1)
            self.smoothing_combo.setEnabled(False)
            
            smoothing_param_layout.addWidget(smoothing_param_label)
            smoothing_param_layout.addWidget(self.smoothing_combo)
            
            post_group_layout.addWidget(self.clean_blobs_check)
            post_group_layout.addWidget(self.smoothing_check)
            post_group_layout.addLayout(smoothing_param_layout)
            tab2_layout.addWidget(post_group)

            # Группа 4: Кастомизация цветов
            color_group = QGroupBox("Управление цветом контуров")
            color_group_layout = QVBoxLayout(color_group)
            
            color_preset_label = QLabel("Предопределенный набор цветов:")
            self.color_preset_combo = NonScrollComboBox()
            self.color_preset_combo.addItems([
                "Классический",
                "Цвета QUANTEC",
                "Яркий неоновый"
            ])
            self.color_preset_combo.currentTextChanged.connect(self.on_color_preset_changed)
            
            color_group_layout.addWidget(color_preset_label)
            color_group_layout.addWidget(self.color_preset_combo)
            tab2_layout.addWidget(color_group)

            # Группа 5: Параметры соединения с сервером AI Contour 🌐
            conn_group = QGroupBox("Соединение с сервером AI Contour 🌐")
            conn_group_layout = QVBoxLayout(conn_group)
            conn_group_layout.setSpacing(10)
            
            lbl_server_url = QLabel("Адрес сервера (IP и Порт):")
            self.server_url_edit = QLineEdit("http://127.0.0.1:8000")
            self.server_url_edit.setPlaceholderText("http://192.168.1.100:8000")
            
            lbl_client_name = QLabel("Идентификатор клиента (Имя ПК):")
            import socket
            try:
                default_client_name = socket.gethostname()
            except Exception:
                default_client_name = "Клиент ПК"
            self.client_name_edit = QLineEdit(default_client_name)
            self.client_name_edit.setPlaceholderText("Введите имя или кабинет...")
            
            self.btn_test_conn = QPushButton("⚡ Проверить подключение")
            self.btn_test_conn.setObjectName("btnAction")
            self.btn_test_conn.clicked.connect(self.test_server_connection)
            
            self.lbl_conn_status = QLabel("Статус: не проверено ⚪")
            self.lbl_conn_status.setStyleSheet("color: #bdc3c7; font-weight: bold;")
            
            conn_group_layout.addWidget(lbl_server_url)
            conn_group_layout.addWidget(self.server_url_edit)
            conn_group_layout.addWidget(lbl_client_name)
            conn_group_layout.addWidget(self.client_name_edit)
            conn_group_layout.addWidget(self.btn_test_conn)
            conn_group_layout.addWidget(self.lbl_conn_status)
            conn_group.setVisible(False)
            
            self.btn_show_conn_settings = QPushButton("🔑 Настройки соединения (Администратор)")
            self.btn_show_conn_settings.setObjectName("btnBrowse")
            
            def toggle_conn_settings():
                if conn_group.isVisible():
                    conn_group.setVisible(False)
                    self.btn_show_conn_settings.setText("🔑 Настройки соединения (Администратор)")
                else:
                    from PyQt6.QtWidgets import QInputDialog, QLineEdit, QMessageBox
                    ok_text, ok = QInputDialog.getText(
                        self, "Доступ ограничен 🔒", "Введите пароль администратора для изменения настроек соединения:", 
                        QLineEdit.EchoMode.Password
                    )
                    if ok and ok_text == "rtp":
                        conn_group.setVisible(True)
                        self.btn_show_conn_settings.setText("🔒 Скрыть настройки соединения")
                    elif ok:
                        QMessageBox.critical(self, "Ошибка доступа ❌", "Неверный пароль!")
            
            self.btn_show_conn_settings.clicked.connect(toggle_conn_settings)
            
            tab2_layout.addWidget(self.btn_show_conn_settings)
            tab2_layout.addWidget(conn_group)

            # Звук в конце
            self.sound_check = QCheckBox("🔔 Звук при завершении автооконтуривания")
            self.sound_check.setChecked(True)
            tab2_layout.addWidget(self.sound_check)
            
            tab2_layout.addStretch()
            tab2_scroll.setWidget(tab2_widget)
            self.tab_widget.addTab(tab2_scroll, "⚙️ Настройки")

            # ------------------------------------------------------------------
            # ВКЛАДКА 3: Статистика автооконтурирований
            # ------------------------------------------------------------------
            tab_stats_scroll = self.create_statistics_tab()
            self.tab_widget.addTab(tab_stats_scroll, "📊 Статистика")

            # ------------------------------------------------------------------
            # ВКЛАДКА 3: Справка и дисклеймер
            # ------------------------------------------------------------------
            tab3_widget = QWidget()
            tab3_layout = QVBoxLayout(tab3_widget)
            tab3_layout.setContentsMargins(0, 0, 0, 0)
            tab3_layout.setSpacing(0)

            from PyQt6.QtWidgets import QTextBrowser
            help_browser = QTextBrowser()
            help_browser.setOpenExternalLinks(True)
            help_browser.setFrameShape(QTextBrowser.Shape.NoFrame)
            help_browser.setHtml("""<!DOCTYPE html>
<html>
<head>
<style>
    body {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
        line-height: 1.6;
        margin: 8px 12px;
        padding: 0;
    }
    h1 {
        color: #ffffff;
        font-size: 17px;
        border-bottom: 2px solid #007acc;
        padding-bottom: 6px;
        margin-top: 8px;
        margin-bottom: 10px;
    }
    h2 {
        color: #007acc;
        font-size: 13px;
        margin-top: 14px;
        margin-bottom: 6px;
        font-weight: bold;
    }
    ul {
        margin: 4px 0;
        padding-left: 18px;
    }
    li {
        margin-bottom: 5px;
    }
    p {
        margin: 6px 0;
    }
    .disclaimer-box {
        background-color: #2c1a1a;
        border: 1px solid #d32f2f;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 14px;
    }
    .disclaimer-title {
        color: #f44336;
        font-weight: bold;
        font-size: 13px;
        margin-bottom: 5px;
    }
    .highlight {
        color: #0098ff;
        font-weight: bold;
    }
    .card {
        background-color: #242424;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 10px;
    }
</style>
</head>
<body>
    <h1>Справка по работе с AI Contour (Краснодар)📖</h1>

    <p><b>AI Contour</b> — интеллектуальное ПО для автоматического сегментирования органов риска (OAR) на КТ-снимках DICOM с использованием нейросети <b>TotalSegmentator</b>.</p>

    <div class="card">
        <h2>Основные возможности 🚀</h2>
        <ul>
            <li><b>Динамические пресеты:</b> Редактируйте анатомические пресеты во внешних файлах в папке <span class="highlight">config/presets/</span>.</li>
            <li><b>GPU-ускорение:</b> При наличии Nvidia CUDA расчёты выполняются в 20–30 раз быстрее.</li>
            <li><b>3D постобработка:</b> Очистка мелкого шума (Remove small blobs) и сглаживание Гаусса.</li>
            <li><b>Кастомизация цветов:</b> Двойной клик по органу — выбор цвета. Палитры QUANTEC и Неон.</li>
            <li><b>Просмотр структур:</b> Включите «Отобразить структуры» на вкладке снимков для наложения контуров на КТ.</li>
            <li><b>Режим слияния:</b> Дополняйте существующий RTSTRUCT или создавайте новый.</li>
        </ul>
    </div>

    <div class="card">
        <h2>Порядок работы 📋</h2>
        <ul>
            <li>На вкладке <b>«⚙️ Настройки»</b> выберите папку с КТ-снимками DICOM и нажмите <b>«📂 Источник»</b>.</li>
            <li>На вкладке <b>«🎯 Контуры»</b> выберите пресет органов и пациента в таблице.</li>
            <li>Настройте режим расчёта (CPU/GPU), точность ИИ и постобработку.</li>
            <li>Нажмите <b>«ОТПРАВИТЬ В ОЧЕРЕДЬ НА СЕРВЕР 🚀»</b> и дождитесь завершения.</li>
            <li>После завершения включите <b>«Отобразить структуры»</b> для просмотра результатов.</li>
        </ul>
    </div>

    <div class="card">
        <h2>Форматы файлов структур 📁</h2>
        <ul>
            <li>Выходной файл RTSTRUCT сохраняется рядом со снимками: <span class="highlight">STR_YYYYMMDD_HHMMSS.dcm</span>.</li>
            <li>Файл совместим с TPS: Monaco, RayStation, Eclipse и другими.</li>
            <li>Поддерживается импорт в OIS системы через стандартный DICOM RT-импорт.</li>
        </ul>
    </div>

    <div class="disclaimer-box">
        <div class="disclaimer-title">⚠️ ВАЖНЫЙ МЕДИЦИНСКИЙ ДИСКЛЕЙМЕР</div>
        <p style="margin: 0; font-size: 12px; color: #e0b0b0;">
            Данное ПО предоставляется исключительно для научных и исследовательских целей (<b>Research Use Only</b>).<br><br>
            Автоматическая разметка <b>не является окончательной клинической разметкой</b>. Любая импортированная разметка
            <b>подлежит обязательному ручному контролю, валидации и коррекции</b> сертифицированным медицинским физиком
            или радиационным онкологом в системе планирования (TPS) перед облучением пациента.
        </p>
    </div>
</body>
</html>""")
            tab3_layout.addWidget(help_browser, 1)
            self.tab_widget.addTab(tab3_widget, "📖 Справка")

            self.splitter.addWidget(self.left_card)

            # --- ПРАВАЯ КОЛОНКА (Терминал логов и управление) ---
            right_card = QFrame()
            right_card.setObjectName("card")
            right_layout = QVBoxLayout(right_card)
            right_layout.setContentsMargins(6, 6, 6, 6)
            right_layout.setSpacing(6)

            logs_header = QLabel("Лог выполнения работы:")
            logs_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.log_edit = QTextEdit()
            self.log_edit.setReadOnly(True)
            self.log_edit.setPlaceholderText("Здесь будет отображаться ход выполнения автооконтурирования...")

            # --- Таблица выбора серии DICOM ---
            table_header = QLabel("Выбор пациента:")
            table_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.series_table = QTableWidget(0, 7)
            self.series_table.setHorizontalHeaderLabels(["ФИО", "ID", "STR", "Область", "Срезы", "Дата КТ", "Путь"])
            self.series_table.setColumnHidden(6, True) # Скрываем путь
            
            header = self.series_table.horizontalHeader()
            # ФИО: Interactive — не сжимается автоматически, пользователь может изменять вручную
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            self.series_table.setColumnWidth(0, 180)  # начальная ширина
            # ID, STR, Срезы, Дата — по содержимому (фиксированная минимальная ширина)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            # Область сканирования: Stretch — сжимается первой при нехватке места
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            # Минимальный размер секций, чтобы данные совсем не пропали
            header.setMinimumSectionSize(32)
            
            self.series_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.series_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.series_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.series_table.customContextMenuRequested.connect(self.show_context_menu)
            self.series_table.cellDoubleClicked.connect(self.on_table_double_clicked)
            self.series_table.setStyleSheet("""
                QTableWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #3c3c3c;
                    border-radius: 4px;
                }
                QHeaderView::section {
                    background-color: #333333;
                    color: #007acc;
                    font-weight: bold;
                    border: 1px solid #2d2d2d;
                }
            """)
            self.series_table.itemSelectionChanged.connect(self.on_series_selected)
            # ----------------------------------

            progress_header = QLabel("Индикатор прогресса:")
            progress_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            queue_header = QLabel("Очередь автооконтурирования (на сервере):")
            queue_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.status_step_label = QLabel("Текущий шаг: Ожидание запуска...")
            self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setFormat("%p%")

            # Метка для отображения ETA (прошедшее время + прогноз окончания)
            self.eta_label = QLabel("")
            self.eta_label.setStyleSheet("color: #888888; font-size: 11px; font-style: italic;")

            self.btn_run = QPushButton("ОТПРАВИТЬ В ОЧЕРЕДЬ НА СЕРВЕР 🚀")
            self.btn_run.setObjectName("btnRun")
            self.btn_run.clicked.connect(self.start_segmentation)

            # Вертикальный сплиттер для главной зоны и зоны логов
            self.v_splitter = QSplitter(Qt.Orientation.Vertical)
            
            # Верхняя панель (Таблица + Вьюер)
            top_panel = QWidget()
            top_layout = QVBoxLayout(top_panel)
            top_layout.setContentsMargins(0, 0, 0, 0)
            
            # Горизонтальный ряд заголовков над сплиттером для экономии высоты
            headers_widget = QWidget()
            headers_layout = QHBoxLayout(headers_widget)
            headers_layout.setContentsMargins(0, 0, 0, 0)
            headers_layout.setSpacing(10)
            
            # Создаем и стилизуем заголовок вьюера
            viewer_section_header = QLabel("Просмотр КТ-снимков:")
            viewer_section_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            
            headers_layout.addWidget(table_header, 6)
            headers_layout.addWidget(viewer_section_header, 4)
            
            top_layout.addWidget(headers_widget)
            
            # --- Вьюер DICOM (PyQtGraph) ---
            viewer_container = QWidget()
            viewer_layout = QVBoxLayout(viewer_container)
            viewer_layout.setContentsMargins(0, 0, 0, 0)
            
            viewer_tools_panel = QFrame()
            viewer_tools_panel.setObjectName("viewerToolsPanel")
            viewer_tools_panel.setStyleSheet("""
                QFrame#viewerToolsPanel {
                    background-color: #1a1a1a;
                    border: 1px solid #2d2d2d;
                    border-left: 4px solid #2ecc71;
                    border-radius: 4px;
                }
                QLabel {
                    color: #888888;
                    font-weight: bold;
                }
            """)
            viewer_tools_layout = QHBoxLayout(viewer_tools_panel)
            viewer_tools_layout.setContentsMargins(10, 6, 10, 6)
            
            self.chk_show_structures = QCheckBox("Отобразить структуры")
            self.chk_show_structures.setEnabled(False)
            self.chk_show_structures.setStyleSheet("""
                QCheckBox {
                    color: #2ecc71;
                    font-weight: bold;
                    font-size: 13px;
                    spacing: 8px;
                }
                QCheckBox:disabled {
                    color: rgba(46, 204, 113, 0.35);
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #2ecc71;
                    border-radius: 4px;
                    background-color: #1a1a1a;
                }
                QCheckBox::indicator:hover {
                    border-color: #27ae60;
                    background-color: #242424;
                }
                QCheckBox::indicator:checked {
                    border-color: #2ecc71;
                    background-color: #2ecc71;
                    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE2LjJ6IiBmaWxsPSIjZmZmZmZmIi8+PC9zdmc+");
                }
                QCheckBox::indicator:disabled {
                    border-color: rgba(46, 204, 113, 0.15);
                    background-color: #1a1a1a;
                }
                QCheckBox::indicator:checked:disabled {
                    border-color: rgba(46, 204, 113, 0.35);
                    background-color: rgba(46, 204, 113, 0.35);
                    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik05IDE2LjJMNC44IDEybC0xLjQgMS40TDkgMTkgMjEgN2wtMS40LTEuNEw5IDE6LjJ6IiBmaWxsPSIjYWFhYWFhIi8+PC9zdmc+");
                }
            """)
            self.chk_show_structures.stateChanged.connect(self.on_show_structures_toggled)
            
            self.rtstruct_combo = NonScrollComboBox()
            self.rtstruct_combo.setEnabled(False)
            self.rtstruct_combo.currentIndexChanged.connect(self.on_viewer_rtstruct_changed)
            
            # Кнопки приближения и отдаления + / -
            self.btn_zoom_in = QPushButton("+")
            self.btn_zoom_in.setToolTip("Приблизить КТ-снимок (Zoom In)")
            self.btn_zoom_in.setFixedWidth(40)
            self.btn_zoom_in.setObjectName("btnBrowse")
            self.btn_zoom_in.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            self.btn_zoom_out = QPushButton("-")
            self.btn_zoom_out.setToolTip("Отдалить КТ-снимок (Zoom Out)")
            self.btn_zoom_out.setFixedWidth(40)
            self.btn_zoom_out.setObjectName("btnBrowse")
            self.btn_zoom_out.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            def zoom_in():
                try:
                    self.dicom_viewer.getView().scaleBy((0.85, 0.85))
                except Exception:
                    pass
            
            def zoom_out():
                try:
                    self.dicom_viewer.getView().scaleBy((1.18, 1.18))
                except Exception:
                    pass
                    
            self.btn_zoom_in.clicked.connect(zoom_in)
            self.btn_zoom_out.clicked.connect(zoom_out)

            viewer_tools_layout.addWidget(self.chk_show_structures)
            viewer_tools_layout.addWidget(QLabel("Файл:"))
            viewer_tools_layout.addWidget(self.rtstruct_combo, 1)
            viewer_tools_layout.addWidget(self.btn_zoom_in)
            viewer_tools_layout.addWidget(self.btn_zoom_out)
            viewer_layout.addWidget(viewer_tools_panel)
            
            self.dicom_viewer = pg.ImageView()
            self.dicom_viewer.ui.roiBtn.hide()
            self.dicom_viewer.ui.menuBtn.hide()
            viewer_layout.addWidget(self.dicom_viewer, 1)
            
            # Связываем сигнал смены кадра с обновлением оверлея
            self.dicom_viewer.sigTimeChanged.connect(self.update_roi_overlay_frame)
            
            # Подключаем кастомный фильтр на viewport() графического виджета
            self.viewer_event_filter = ViewerEventFilter(self.dicom_viewer)
            self.dicom_viewer.ui.graphicsView.viewport().installEventFilter(self.viewer_event_filter)
            
            # Горизонтальный сплиттер: 60% таблица / 40% вьюер
            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.main_splitter.addWidget(self.series_table)
            self.main_splitter.addWidget(viewer_container)
            self.main_splitter.setStretchFactor(0, 6)
            self.main_splitter.setStretchFactor(1, 4)
            self.main_splitter.setSizes([600, 400])
            
            top_layout.addWidget(self.main_splitter, 1)  # stretch=1: main_splitter занимает всё оставшееся пространство
            
            # Нижняя панель (Логи + Очередь + Прогресс)
            bottom_panel = QWidget()
            bottom_layout = QVBoxLayout(bottom_panel)
            bottom_layout.setContentsMargins(0, 0, 0, 0)
            bottom_layout.setSpacing(6)
            
            bottom_splitter = QSplitter(Qt.Orientation.Vertical)
            bottom_splitter.setObjectName("bottomSplitter")
            
            # 1. Зона логов
            logs_widget = QWidget()
            logs_layout = QVBoxLayout(logs_widget)
            logs_layout.setContentsMargins(0, 0, 0, 0)
            logs_layout.setSpacing(4)
            logs_layout.addWidget(logs_header)
            logs_layout.addWidget(self.log_edit, 1)
            
            # 2. Зона очереди работ (с заголовком)
            queue_widget = QWidget()
            queue_layout = QVBoxLayout(queue_widget)
            queue_layout.setContentsMargins(0, 0, 0, 0)
            queue_layout.setSpacing(4)
            queue_layout.addWidget(queue_header)

            # Инициализация и добавление компактной таблицы очереди
            self.table_queue = QueueTableWidget(self)
            self.table_queue.parent_app = self
            self.table_queue.setColumnCount(7)
            self.table_queue.setHorizontalHeaderLabels([
                "Клиент", "Пациент", "ID Пациента", "Пресет", "Статус", "Прогресс", "Получено"
            ])
            self.table_queue.setMinimumHeight(95)  # Идеальный минимальный компактный размер для шапки + 2 строк!
            
            header = self.table_queue.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            
            self.table_queue.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table_queue.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table_queue.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table_queue.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.table_queue.customContextMenuRequested.connect(self.show_context_menu)

            queue_layout.addWidget(self.table_queue, 1)
            
            bottom_splitter.addWidget(logs_widget)
            bottom_splitter.addWidget(queue_widget)
            bottom_splitter.setStretchFactor(0, 1)
            bottom_splitter.setStretchFactor(1, 1)
            bottom_splitter.setSizes([180, 120])
            
            # Добавляем сплиттер в главный лейаут нижней панели
            bottom_layout.addWidget(bottom_splitter, 1)
            
            # 3. Элементы прогресса и кнопка запуска (идут под сплиттером, всегда на виду)
            bottom_layout.addWidget(progress_header)
            bottom_layout.addWidget(self.status_step_label)
            bottom_layout.addWidget(self.progress_bar)
            bottom_layout.addWidget(self.eta_label)
            bottom_layout.addWidget(self.btn_run)
            
            self.v_splitter.addWidget(top_panel)
            self.v_splitter.addWidget(bottom_panel)
            # Вертикальный сплиттер: 50% верхняя зона / 50% логи
            self.v_splitter.setStretchFactor(0, 1)
            self.v_splitter.setStretchFactor(1, 1)
            self.v_splitter.setSizes([500, 500])
            
            right_layout.addWidget(self.v_splitter, 1)  # stretch=1: v_splitter заполняет right_card

            self.splitter.addWidget(right_card)
            self.splitter.setStretchFactor(0, 0)
            self.splitter.setStretchFactor(1, 1)

            # Псевдонимы для совместимости с требованиями ТЗ
            self.palette_combo = self.color_preset_combo
            self.structures_list = self.organs_list

            # Инициализация списков пресетов и органов из конфигурации config/ движка
            self.init_presets_and_organs()
            self.update_license_status_label()

            # Установка палитры по умолчанию на Цвета QUANTEC
            self.palette_combo.setCurrentText("Цвета QUANTEC")

            # Подключаем сохранение настроек
            self.sound_check.stateChanged.connect(self.on_sound_check_changed)
            self.clean_blobs_check.stateChanged.connect(self.save_settings)
            self.smoothing_check.stateChanged.connect(self.save_settings)
            self.precision_combo.currentIndexChanged.connect(self.save_settings)
            self.smoothing_combo.currentIndexChanged.connect(self.save_settings)
            self.color_preset_combo.currentIndexChanged.connect(self.save_settings)
            
            self.splitter.setSizes([490, 430])

        def resizeEvent(self, event):
            """Динамическое ограничение максимальной ширины левой панели до 50% ширины окна."""
            super().resizeEvent(event)
            if hasattr(self, 'left_card'):
                max_w = max(490, int(self.width() * 0.5))
                self.left_card.setMaximumWidth(max_w)

        def update_license_status_label(self):
            """Обновляет статус лицензии (заглушка на клиенте)."""
            pass

        def show_licenses_dialog(self):
            """Открывает окно управления лицензиями суб-моделей ИИ."""
            dialog = LicensesDialog(self, self.engine)
            dialog.exec()
            # После закрытия диалога обновляем статус-лейбл и список органов
            self.update_license_status_label()
            self.init_presets_and_organs()

        def show_models_dialog(self):
            """Открывает окно управления моделями ИИ."""
            dialog = ModelsDialog(self, self.engine)
            dialog.exec()

        def on_sound_check_changed(self):
            self.save_settings()
            if self.sound_check.isChecked():
                try:
                    import winsound
                    winsound.Beep(523, 150)
                except Exception:
                    pass

        def init_presets_and_organs(self):
            """Инициализирует комбобокс пресетов и список органов из конфигурации config/."""
            self.is_updating_presets = True
            self.preset_combo.clear()
            self.organs_list.clear()

            # Первый элемент — пустая строка-подсказка (ничего не выделяет)
            self.preset_combo.addItem("— Выберите пресет —")
            # Добавляем пресеты из движка
            presets_keys = list(self.engine.presets.keys())
            self.preset_combo.addItems(presets_keys)
            self.preset_combo.addItem("Пользовательский (Custom)")

            # Использование глобального ORGAN_GROUPS из config.py

            # Получаем все доступные органы динамически из движка
            all_supported_organs = self.engine.get_all_supported_organs()
            placed_organs = set()

            for group_title, organs in ORGAN_GROUPS.items():
                is_collapsed = self.collapsed_groups.get(group_title, False)
                prefix = "[+] " if is_collapsed else "[-] "
                header_item = QListWidgetItem(f"{prefix}{group_title} ({len(organs)})")
                header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                header_item.setCheckState(Qt.CheckState.Unchecked)
                header_item.setData(Qt.ItemDataRole.UserRole, "header")
                
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                header_item.setForeground(QBrush(QColor("#007acc")))
                header_item.setBackground(QBrush(QColor("#242424")))
                self.organs_list.addItem(header_item)

                for org in organs:
                    placed_organs.add(org)
                    # Проверяем, есть ли такой орган в ru_names
                    ru_name = self.engine.ru_names.get(org, org)
                    
                    # Проверка лицензии для суб-моделей
                    task = ROI_TO_TASK_MAP.get(org, 'total')
                    is_licensed_task = task in LICENSED_TASKS
                    has_license = hasattr(self.engine, "licenses") and isinstance(self.engine.licenses, str) and self.engine.licenses.strip()
                    
                    if is_licensed_task and not has_license:
                        item = QListWidgetItem(f"   [🔒] {ru_name} (Нужна лицензия)")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable & ~Qt.ItemFlag.ItemIsEnabled)
                        item.setForeground(QBrush(QColor("#777777")))
                    else:
                        item = QListWidgetItem(f"   {ru_name}")
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Unchecked)
                        
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    
                    # Установка цветного квадратика-иконки для OAR
                    self.update_item_color_icon(item, org)
                    
                    if is_collapsed:
                        item.setHidden(True)
                    
                    self.organs_list.addItem(item)
            
            # Исключаем дубликаты с похожими названиями, чтобы они не засоряли раздел "Остальное"
            duplicates_to_exclude = {
                "brainstem", "eye_lens_left", "eye_lens_right", "iliac_vena_left", "iliac_vena_right",
                "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
                "kidney_cyst_left", "kidney_cyst_right", "thalamus", "caudate_nucleus", "lentiform_nucleus", "ventricle",
                "heart_myocardium", "heart_atrium_left", "heart_atrium_right", "heart_ventricle_left", "heart_ventricle_right"
            }

            # Функция фильтрации: оставляем только ребра, позвонки (C1-L5) и глубокие мышцы
            def is_allowed_other(org):
                if org.startswith("rib_"):
                    return True
                if org.startswith("vertebrae_"):
                    m = re.match(r"^vertebrae_([cCtTlL])(\d+)$", org)
                    if m:
                        return True
                    return False
                if org in ["autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right"]:
                    return True
                return False

            other_organs = [
                org for org in all_supported_organs 
                if org not in placed_organs 
                and org not in duplicates_to_exclude 
                and is_allowed_other(org)
            ]
            
            if other_organs:
                is_other_collapsed = self.collapsed_groups.get("Остальное", True)
                prefix = "[+] " if is_other_collapsed else "[-] "
                other_header = QListWidgetItem(f"{prefix}━━━ ОСТАЛЬНОЕ ━━━ ({len(other_organs)})")
                other_header.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                other_header.setCheckState(Qt.CheckState.Unchecked)
                other_header.setData(Qt.ItemDataRole.UserRole, "header")
                font = other_header.font()
                font.setBold(True)
                other_header.setFont(font)
                other_header.setForeground(QBrush(QColor("#007acc")))
                other_header.setBackground(QBrush(QColor("#242424")))
                self.organs_list.addItem(other_header)
                
                for org in other_organs:
                    ru_name = self.engine.ru_names.get(org, org)
                    
                    # Проверка лицензии для суб-моделей
                    task = ROI_TO_TASK_MAP.get(org, 'total')
                    is_licensed_task = task in LICENSED_TASKS
                    has_license = hasattr(self.engine, "licenses") and isinstance(self.engine.licenses, str) and self.engine.licenses.strip()
                    
                    if is_licensed_task and not has_license:
                        item = QListWidgetItem(f"   [🔒] {ru_name} (Нужна лицензия)")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable & ~Qt.ItemFlag.ItemIsEnabled)
                        item.setForeground(QBrush(QColor("#777777")))
                    else:
                        item = QListWidgetItem(f"   {ru_name}")
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        item.setCheckState(Qt.CheckState.Unchecked)
                        
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    self.update_item_color_icon(item, org)
                    
                    if is_other_collapsed:
                        item.setHidden(True)
                        
                    self.organs_list.addItem(item)

            self.is_updating_presets = False
            self.update_checked_organs_count()

        def update_item_color_icon(self, item: QListWidgetItem, organ_name: str):
            """Генерирует и устанавливает цветную иконку для органа в списке."""
            pixmap = QPixmap(14, 14)
            if isinstance(organ_name, dict):
                org_str = organ_name.get("name") or (list(organ_name.keys())[0] if organ_name else "")
            else:
                org_str = organ_name
            
            if not isinstance(org_str, str):
                org_str = str(org_str) if org_str else ""
                
            color_rgb = self.engine.colors.get(org_str, [128, 128, 128])
            pixmap.fill(QColor(color_rgb[0], color_rgb[1], color_rgb[2]))
            item.setIcon(QIcon(pixmap))

        def update_organs_list_highlighting(self):
            """
            Подсвечивает бирюзовым цветом (#00ffd0) и делает полужирными
            те органы в списке, которые присутствуют в текущем загруженном RTSTRUCT.
            Отсутствующие органы затеняет серым цветом (#777777).
            Если галочка 'Отобразить структуры' выключена, возвращает белый цвет (#ffffff).
            """
            from PyQt6.QtGui import QBrush, QColor
            from PyQt6.QtCore import Qt
            import re
            
            is_highlight_active = getattr(self, 'chk_show_structures', None) and self.chk_show_structures.isChecked()
            has_rtstruct = getattr(self, '_cached_rtstruct', None) is not None
            
            self.organs_list.blockSignals(True)
            try:
                file_organs = set()
                if is_highlight_active and has_rtstruct:
                    roi_names = self._cached_rtstruct.get_roi_names()
                    normalize_name = lambda n: re.sub(r'[^a-z0-9]', '', n.lower())
                    all_supported_organs = self.engine.get_all_supported_organs()
                    
                    supported_norm_map = {}
                    for org in all_supported_organs:
                        supported_norm_map[normalize_name(org)] = org
                        mon_pretty = self.engine.get_monaco_pretty_name(org)
                        if mon_pretty:
                            supported_norm_map[normalize_name(mon_pretty)] = org

                    def get_mapped_organ(roi_name_str: str) -> str:
                        norm = normalize_name(roi_name_str)
                        if norm in EXTERNAL_ALIASES:
                            return EXTERNAL_ALIASES[norm]
                        if norm in supported_norm_map:
                            return supported_norm_map[norm]
                        return roi_name_str.lower().replace(" ", "_")

                    file_organs = set(get_mapped_organ(r) for r in roi_names)

                for i in range(self.organs_list.count()):
                    item = self.organs_list.item(i)
                    itm_data = item.data(Qt.ItemDataRole.UserRole)
                    if itm_data != "header" and itm_data:
                        if isinstance(itm_data, dict):
                            org_name = itm_data.get("name") or (list(itm_data.keys())[0] if itm_data else "")
                        else:
                            org_name = itm_data
                            
                        if not org_name:
                            continue
                            
                        font = item.font()
                        if is_highlight_active and has_rtstruct:
                            mapped = get_mapped_organ(org_name)
                            if mapped in file_organs:
                                item.setForeground(QBrush(QColor("#00ffd0")))
                                font.setBold(True)
                            else:
                                item.setForeground(QBrush(QColor("#777777")))
                                font.setBold(False)
                        else:
                            item.setForeground(QBrush(QColor("#ffffff")))
                            font.setBold(False)
                        item.setFont(font)
            except Exception as e:
                logger.error(f"Ошибка при обновлении подсветки списка органов: {e}")
            finally:
                self.organs_list.blockSignals(False)

        def load_settings(self):
            """Загружает сохраненное состояние интерфейса."""
            self.preset_combo.blockSignals(True)
            self.organs_list.blockSignals(True)
            self.is_updating_presets = True
            
            try:
                input_dir = self.settings.value("input_dir", "")
                if input_dir:
                    self.input_edit.setText(input_dir)
                self.last_alternative_output_dir = self.settings.value("alternative_output_dir", "")

                # Параметры соединения с сервером
                server_url = self.settings.value("server_url", "http://127.0.0.1:8000")
                self.server_url_edit.setText(server_url)
                
                client_name = self.settings.value("client_name", "")
                if client_name:
                    self.client_name_edit.setText(client_name)

                # При старте комбобокс ВСЕГДА в положении заглушки.
                # Сохранённое имя пресета используется только для подбора текста в комбо
                # ПОСЛЕ того как органы восстановлены из checked_organs.
                self.preset_combo.setCurrentIndex(0)
                # Убираем устаревший ключ «preset» из настроек — теперь состояние
                # определяется исключительно по checked_organs.
                self.settings.remove("preset")

                # Доп параметры постобработки и точности
                precision_idx = self.settings.value("precision_mode", 0, type=int)
                self.precision_combo.setCurrentIndex(precision_idx)

                clean_blobs = self.settings.value("clean_blobs", True, type=bool)
                self.clean_blobs_check.setChecked(clean_blobs)

                smoothing = self.settings.value("smoothing", False, type=bool)
                self.smoothing_check.setChecked(smoothing)
                self.smoothing_combo.setEnabled(smoothing)

                smoothing_idx = self.settings.value("smoothing_idx", 1, type=int)
                self.smoothing_combo.setCurrentIndex(smoothing_idx)

                color_preset = self.settings.value("color_preset", "Цвета QUANTEC")
                self.color_preset_combo.setCurrentText(color_preset)

                play_sound = self.settings.value("play_sound", True, type=bool)
                self.sound_check.setChecked(play_sound)

                # Загружаем выбранные ресурсы
                use_gpu = self.settings.value("use_gpu", True, type=bool)
                self.radio_gpu.setChecked(use_gpu)
                self.radio_cpu.setChecked(not use_gpu)

                # Восстанавливаем галочки органов (без сигналов)
                checked_organs = self.settings.value("checked_organs", None)
                if checked_organs is not None:
                    if not isinstance(checked_organs, list):
                        checked_organs = [checked_organs]
                    self.organs_list.blockSignals(True)
                    try:
                        for i in range(self.organs_list.count()):
                            item = self.organs_list.item(i)
                            organ_name = item.data(Qt.ItemDataRole.UserRole)
                            if organ_name == "header":
                                continue
                            item.setCheckState(
                                Qt.CheckState.Checked if organ_name in checked_organs
                                else Qt.CheckState.Unchecked
                            )
                    finally:
                        self.organs_list.blockSignals(False)

                # Обновляем состояния заголовков категорий
                self.update_headers_check_states()
                # НЕ вызываем _sync_preset_combo_to_organs — при старте
                # комбобокс всегда должен показывать заглушку.
            finally:
                self.is_updating_presets = False
                self.organs_list.blockSignals(False)
                self.preset_combo.blockSignals(False)
                
                input_dir = self.settings.value("input_dir", "")
                if input_dir and os.path.isdir(input_dir):
                    self.start_dicom_scan(input_dir, is_manual=True)
                    self.scan_timer.start()
                self.update_checked_organs_count()

        def save_settings(self):
            """Сохраняет состояние интерфейса в QSettings."""
            self.settings.setValue("input_dir", self.input_edit.text().strip())
            self.settings.setValue("server_url", self.server_url_edit.text().strip())
            self.settings.setValue("client_name", self.client_name_edit.text().strip())
            if hasattr(self, "last_alternative_output_dir") and self.last_alternative_output_dir:
                self.settings.setValue("alternative_output_dir", self.last_alternative_output_dir)
            current_preset = self.preset_combo.currentText()
            if current_preset != "— Выберите пресет —":
                self.settings.setValue("preset", current_preset)
            else:
                self.settings.remove("preset")
            self.settings.setValue("precision_mode", self.precision_combo.currentIndex())
            self.settings.setValue("clean_blobs", self.clean_blobs_check.isChecked())
            self.settings.setValue("smoothing", self.smoothing_check.isChecked())
            self.settings.setValue("smoothing_idx", self.smoothing_combo.currentIndex())
            self.settings.setValue("color_preset", self.color_preset_combo.currentText())
            self.settings.setValue("play_sound", self.sound_check.isChecked())
            self.settings.setValue("use_gpu", self.radio_gpu.isChecked())
            
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

        def test_server_connection(self):
            """Проверяет подключение к указанному FastAPI серверу."""
            import requests
            server_url = self.server_url_edit.text().strip()
            if not server_url:
                QMessageBox.warning(self, "Ошибка", "Укажите адрес сервера!")
                return
                
            self.lbl_conn_status.setText("Проверка связи...")
            self.lbl_conn_status.setStyleSheet("color: #f1c40f; font-weight: bold;")
            QApplication.processEvents() # Обновить UI
            
            try:
                # Отсылаем GET запрос на корень API
                r = requests.get(server_url, timeout=5)
                if r.status_code == 200:
                    self.lbl_conn_status.setText("Статус: подключено 🟢")
                    self.lbl_conn_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
                    QMessageBox.information(
                        self, "Успех", 
                        f"Успешное соединение с сервером AI Contour!\n\n"
                        f"Версия API: {r.json().get('version', 'Неизвестно')}\n"
                        f"Статус паузы очереди: {'Приостановлена' if r.json().get('is_paused') else 'Активна'}"
                    )
                else:
                    self.lbl_conn_status.setText("Статус: ошибка API 🔴")
                    self.lbl_conn_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
                    QMessageBox.warning(self, "Ошибка соединения", f"Сервер ответил кодом {r.status_code}.")
            except Exception as e:
                self.lbl_conn_status.setText("Статус: нет связи 🔴")
                self.lbl_conn_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
                QMessageBox.critical(self, "Сбой подключения", f"Не удалось связаться с сервером:\n{e}")

        def select_input_dir(self):
            dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку с КТ-снимками DICOM")
            if dir_path:
                self.input_edit.setText(dir_path)
                self.save_settings()
                self.start_dicom_scan(dir_path, is_manual=True)
                self.scan_timer.start()

        def start_dicom_scan(self, dir_path: str, is_manual: bool = True):
            if hasattr(self, 'scan_worker') and self.scan_worker.isRunning():
                return
                
            self.btn_run.setEnabled(False)
            if is_manual:
                self.btn_run.setText("СКАНИРОВАНИЕ ПАПОК...")
            
            self.scan_worker = DicomScanWorker(dir_path, is_manual_scan=is_manual)
            self.scan_worker.scan_started.connect(self.on_scan_started)
            self.scan_worker.scan_progress.connect(self.on_scan_progress)
            self.scan_worker.scan_completed.connect(self.on_scan_completed)
            self.scan_worker.error_signal.connect(lambda e: logging.getLogger("ContourEngine").error(f"Ошибка сканирования: {e}"))
            self.scan_worker.start()

        def start_background_scan(self):
            dir_path = self.input_edit.text().strip()
            if dir_path and os.path.isdir(dir_path):
                self.start_dicom_scan(dir_path, is_manual=False)

        def on_scan_started(self, total_dcm, total_dirs, is_manual):
            if is_manual and total_dcm > 500:
                self.progress_dialog = QProgressDialog("Сканирование папок DICOM...", None, 0, total_dirs, self)
                self.progress_dialog.setWindowTitle("Поиск исследований")
                self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                self.progress_dialog.setMinimumDuration(0)
                self.progress_dialog.setValue(0)
                self.progress_dialog.setCancelButton(None)
            else:
                self.progress_dialog = None

        def on_scan_progress(self, val):
            if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                self.progress_dialog.setValue(val)

        def on_scan_completed(self, results):
            self._is_updating_table = True
            try:
                if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
                    self.progress_dialog.close()
                    self.progress_dialog = None
                    
                selected_study_path = None
                if self.series_table.selectedItems():
                    selected_row = self.series_table.selectedItems()[0].row()
                    if selected_row >= 0:
                        path_item = self.series_table.item(selected_row, 6)
                        if path_item:
                            selected_study_path = path_item.text()
                    
                self.series_table.setUpdatesEnabled(False)
                
                existing_paths = {}
                for row in range(self.series_table.rowCount()):
                    path_item = self.series_table.item(row, 6)
                    if path_item:
                        existing_paths[path_item.text()] = row
                        
                new_paths = [res[6] for res in results]
                
                rows_to_remove = []
                for path, row in existing_paths.items():
                    if path not in new_paths:
                        rows_to_remove.append(row)
                        
                for row in sorted(rows_to_remove, reverse=True):
                    self.series_table.removeRow(row)
                    
                existing_paths = {}
                for row in range(self.series_table.rowCount()):
                    path_item = self.series_table.item(row, 6)
                    if path_item:
                        existing_paths[path_item.text()] = row
                        
                self.series_table.setSortingEnabled(False)
                
                for (p_name, p_id, str_status, body_part, slice_count, s_date, path) in results:
                    if path in existing_paths:
                        row = existing_paths[path]
                        item_str = self.series_table.item(row, 2)
                        item_slices = self.series_table.item(row, 4)
                        if item_str:
                            item_str.setText(str_status)
                        if item_slices:
                            item_slices.setText(str(slice_count))
                        continue

                    row = self.series_table.rowCount()
                    self.series_table.insertRow(row)
                    
                    self.series_table.setItem(row, 0, QTableWidgetItem(p_name))
                    
                    item_id = QTableWidgetItem(p_id)
                    item_id.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.series_table.setItem(row, 1, item_id)
                    
                    item_str = QTableWidgetItem(str_status)
                    item_str.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.series_table.setItem(row, 2, item_str)
                    
                    item_bp = QTableWidgetItem(body_part)
                    item_bp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.series_table.setItem(row, 3, item_bp)
                    
                    item_slices = QTableWidgetItem(str(slice_count))
                    item_slices.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.series_table.setItem(row, 4, item_slices)
                    
                    item_date = QTableWidgetItem(s_date)
                    item_date.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.series_table.setItem(row, 5, item_date)
                    
                    self.series_table.setItem(row, 6, QTableWidgetItem(path))
                
                self.series_table.setSortingEnabled(True)
                self.series_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
                
                # Восстанавливаем выделение
                if selected_study_path:
                    target_row = -1
                    for r in range(self.series_table.rowCount()):
                        item_path = self.series_table.item(r, 6)
                        if item_path and item_path.text() == selected_study_path:
                            target_row = r
                            break
                    if target_row >= 0:
                        self.series_table.setCurrentCell(target_row, 0)
                        self.series_table.selectRow(target_row)
                    else:
                        self.series_table.clearSelection()
            except Exception as e:
                logger.error(f"Ошибка при обновлении таблицы исследований: {e}")
            finally:
                self.series_table.setUpdatesEnabled(True)
                self.on_scan_finished()
                self._is_updating_table = False
            
        def update_run_button(self, is_patient_selected: bool, custom_text: str = None):
            if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
                target_text = "ОТМЕНИТЬ РАСЧЕТ ❌"
                target_enabled = True
                if self.btn_run.text() != target_text:
                    self.btn_run.setText(target_text)
                if self.btn_run.isEnabled() != target_enabled:
                    self.btn_run.setEnabled(target_enabled)
                self.btn_run.setStyleSheet("""
                    QPushButton#btnRun {
                        background-color: #c0392b;
                        color: white;
                        font-weight: bold;
                        border: 1px solid #962d22;
                        font-size: 14px;
                        padding: 12px;
                        border-radius: 6px;
                    }
                    QPushButton#btnRun:hover {
                        background-color: #e74c3c;
                        border: 1px solid #c0392b;
                    }
                    QPushButton#btnRun:pressed {
                        background-color: #962d22;
                    }
                """)
                return

            if getattr(self, 'chk_show_structures', None) and self.chk_show_structures.isChecked():
                target_text = "ВЫЙТИ ИЗ РЕЖИМА ПРОСМОТРА"
                target_enabled = True
            else:
                target_text = custom_text if custom_text else ("ОТПРАВИТЬ В ОЧЕРЕДЬ НА СЕРВЕР 🚀" if is_patient_selected else "ВЫБЕРИТЕ ПАЦИЕНТА В ТАБЛИЦЕ")
                target_enabled = is_patient_selected if custom_text != "КТ-СЕРИИ НЕ НАЙДЕНЫ" else False
            
            if self.btn_run.text() != target_text:
                self.btn_run.setText(target_text)
                
            if self.btn_run.isEnabled() != target_enabled:
                self.btn_run.setEnabled(target_enabled)
                
            current_style = self.btn_run.styleSheet()
            if getattr(self, 'chk_show_structures', None) and self.chk_show_structures.isChecked():
                self.btn_run.setStyleSheet("background-color: #d87a00; color: white; font-weight: bold; border-radius: 4px; border: none;")
            else:
                if is_patient_selected and "background-color: #0078d7" not in current_style:
                    self.btn_run.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
                elif not is_patient_selected and current_style != "":
                    self.btn_run.setStyleSheet("")

        def on_scan_finished(self):
            if self.series_table.rowCount() == 0:
                self.update_run_button(False, "КТ-СЕРИИ НЕ НАЙДЕНЫ")
            else:
                if self.series_table.selectedItems():
                    self.on_series_selected()
                else:
                    self.update_run_button(False, "ВЫБЕРИТЕ ПАЦИЕНТА В ТАБЛИЦЕ")
                
        def on_series_selected(self):
            selected = self.series_table.selectedItems()
            if selected:
                self.update_run_button(True, "ОТПРАВИТЬ В ОЧЕРЕДЬ НА СЕРВЕР 🚀")
                row = selected[0].row()
                selected_path = self.series_table.item(row, 6).text()
                
                # Меняем UI и обновляем вьюер ТОЛЬКО при ручном клике пользователя
                if not getattr(self, "_is_updating_table", False):
                    # Автоматический выбор пресета на основе области сканирования
                    area_item = self.series_table.item(row, 3)
                    area_text = area_item.text().strip() if area_item else ""
                    if area_text:
                        area_lower = area_text.lower()
                        matched_preset = None
                        
                        if any(x in area_lower for x in ["head", "neck", "голова", "шея", "brain", "мозг", "larynx", "гортан"]):
                            matched_preset = "Голова и шея (Head & Neck)"
                        elif any(x in area_lower for x in ["thorax", "chest", "lung", "груд", "легк", "кост", "rib"]):
                            matched_preset = "Грудная клетка (Thorax)"
                        elif any(x in area_lower for x in ["pelvis", "prostate", "bladder", "таз", "мочевой", "простат"]):
                            matched_preset = "Малый таз (Pelvis)"
                            
                        if matched_preset:
                            # Ищем индекс пресета в комбобоксе
                            idx = self.preset_combo.findText(matched_preset)
                            if idx >= 0 and self.preset_combo.currentText() != matched_preset:
                                logger.info(f"Автоматически выбран пресет '{matched_preset}' на основе области сканирования: '{area_text}'")
                                self.preset_combo.blockSignals(True)
                                self.preset_combo.setCurrentIndex(idx)
                                self.preset_combo.blockSignals(False)
                                # Применяем пресет
                                self.apply_preset_checked_states(matched_preset)
                    
                    # Фоновый поиск реального пути файла (теперь не моргает от таймера)
                    self.check_for_rtstruct(selected_path)
                    
                    self.update_viewer_with_dicom(selected_path)
                    
                    str_status = self.series_table.item(row, 2).text()
                    if str_status == "Нет" or str_status == "No":
                        self.status_rtstruct_label.setText("RTSTRUCT: не найден (будет создан новый)")
                        self.status_rtstruct_label.setStyleSheet("color: #888888;")
                        self.radio_merge_merge.setEnabled(False)
                        self.update_merge_combo_state()
                        self.radio_merge_new.setChecked(True)
                    else:
                        # Если хотим вывести имя файла, можно взять из self.existing_rtstruct_path
                        if self.existing_rtstruct_path:
                            basename = os.path.basename(self.existing_rtstruct_path)
                            self.status_rtstruct_label.setText(f"RTSTRUCT: обнаружен {basename}")
                        else:
                            self.status_rtstruct_label.setText("RTSTRUCT: обнаружен")
                        self.status_rtstruct_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
                        self.radio_merge_merge.setEnabled(True)
                        self.update_merge_combo_state()

        def on_table_double_clicked(self, row, col):
            path = self.series_table.item(row, 6).text()
            if path and os.path.exists(path):
                try:
                    os.startfile(os.path.normpath(path))
                except Exception as e:
                    logger.error(f"Не удалось открыть папку: {e}")

        def show_context_menu(self, position):
            selected = self.series_table.selectedItems()
            if not selected:
                return
            
            row = selected[0].row()
            path = self.series_table.item(row, 6).text()
            
            menu = QMenu(self.series_table)
            delete_action = menu.addAction("Удалить")
            
            action = menu.exec(self.series_table.viewport().mapToGlobal(position))
            if action == delete_action:
                self.scan_timer.stop()
                reply = QMessageBox.question(
                    self,
                    "Удаление исследования",
                    "Вы уверены, что хотите безвозвратно удалить папку этого исследования с диска?\n\n" + path,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        import shutil
                        shutil.rmtree(path)
                        self.series_table.removeRow(row)
                        logger.info(f"Папка {path} безвозвратно удалена.")
                    except Exception as e:
                        QMessageBox.critical(self, "Ошибка удаления", f"Не удалось удалить папку:\n{e}")
                        logger.error(f"Ошибка удаления: {e}")
                self.scan_timer.start(15000)

        def update_viewer_with_dicom(self, folder_path: str):
            if not folder_path or not os.path.isdir(folder_path):
                return
            try:
                import pydicom
                import numpy as np
                import glob
                
                dcm_files = glob.glob(os.path.join(folder_path, "*.dcm"))
                slices = []
                for f in dcm_files:
                    try:
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        if hasattr(ds, "Modality") and ds.Modality == "CT" and hasattr(ds, "ImagePositionPatient"):
                            ds = pydicom.dcmread(f)
                            slices.append(ds)
                    except Exception:
                        pass
                
                if not slices:
                    return
                    
                # Сортировка по Z
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                
                # Формирование 3D массива с учетом HU (RescaleSlope/Intercept)
                volume = []
                for s in slices:
                    image = s.pixel_array.astype(np.float32)
                    slope = getattr(s, 'RescaleSlope', 1)
                    intercept = getattr(s, 'RescaleIntercept', 0)
                    image = image * slope + intercept
                    volume.append(image)
                    
                volume_3d = np.stack(volume)
                # Коррекция Window Level (Контраста)
                volume_3d = np.clip(volume_3d, -160, 240)
                
                # Транспонирование для правильной ориентации в pyqtgraph
                volume_3d = np.transpose(volume_3d, (0, 2, 1))
                
                self.volume_3d_base = volume_3d
                self.current_dicom_dir = folder_path
                # Сохраняем Z-позиции срезов для корректного slice-by-slice маппинга контуров
                self.z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
                self.dicom_pixel_spacing = (
                    float(getattr(slices[0], 'PixelSpacing', [1, 1])[0]),
                    float(getattr(slices[0], 'PixelSpacing', [1, 1])[1])
                )
                self.dicom_image_position = [
                    float(slices[0].ImagePositionPatient[0]),
                    float(slices[0].ImagePositionPatient[1]),
                ]
                self.dicom_viewer.setImage(self.volume_3d_base)
                
                # Принудительно вызываем обновление оверлея если галка включена
                if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                    self.on_show_structures_changed()
            except Exception as e:
                logger.warning(f"Не удалось загрузить DICOM во вьюер: {e}")

        def _clear_imported_organs(self):
            """Удаляет все динамически импортированные сторонние органы из списка."""
            self.organs_list.blockSignals(True)
            try:
                i = self.organs_list.count() - 1
                while i >= 0:
                    item = self.organs_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole + 1) is True:
                        self.organs_list.takeItem(i)
                    i -= 1
                self.imported_items = []
            except Exception as e:
                logger.warning(f"Ошибка при очистке импортированных органов: {e}")
            finally:
                self.organs_list.blockSignals(False)

        def check_for_rtstruct(self, directory: str):
            """Находит все RTSTRUCT файлы в выбранной папке."""
            self.existing_rtstruct_path = None
            self._last_loaded_rtstruct = None
            self._loaded_roi_masks = {}
            self._cached_rtstruct = None
            self._cached_rtstruct_path = None
            
            if hasattr(self, 'rtstruct_combo'):
                self.rtstruct_combo.blockSignals(True)
                self.rtstruct_combo.clear()
                
                # Снимаем галочку и отключаем её
                self.chk_show_structures.blockSignals(True)
                self.chk_show_structures.setChecked(False)
                self.chk_show_structures.setEnabled(False)
                self.chk_show_structures.blockSignals(False)
                
                self.rtstruct_combo.setEnabled(False)
                self.rtstruct_combo.blockSignals(False)
                
            if hasattr(self, 'merge_rtstruct_combo'):
                self.merge_rtstruct_combo.blockSignals(True)
                self.merge_rtstruct_combo.clear()
                self.merge_rtstruct_combo.setEnabled(False)
                self.merge_rtstruct_combo.blockSignals(False)
                
                # Принудительно очищаем старый оверлей из вьюера полностью
                self._clear_roi_overlay(permanent=True)
                self._clear_imported_organs()
                
                # Принудительно возвращаем видимость всем стандартным органам
                self.organs_list.blockSignals(True)
                for i in range(self.organs_list.count()):
                    self.organs_list.item(i).setHidden(False)
                self.organs_list.blockSignals(False)
                
                # Возвращаем левую панель и сплиттеры к стандартным размерам
                if hasattr(self, 'left_card') and hasattr(self, 'splitter'):
                    self.left_card.setMinimumWidth(490)
                    self.left_card.setMaximumWidth(max(490, int(self.width() * 0.5)))
                    self.splitter.setSizes([490, 430])
                if hasattr(self, 'main_splitter'):
                    self.main_splitter.setSizes([600, 400])
                if hasattr(self, 'v_splitter'):
                    self.v_splitter.setSizes([500, 500])
                if hasattr(self, 'dicom_viewer'):
                    try:
                        self.dicom_viewer.getView().autoRange()
                    except Exception:
                        pass
                
                # Принудительно сбрасываем подсветку списка органов
                self.update_organs_list_highlighting()
            
            self.rtstruct_files = []
            if not directory or not os.path.isdir(directory):
                return

            try:
                import pydicom
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        try:
                            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                            if getattr(ds, "Modality", None) == "RTSTRUCT":
                                self.rtstruct_files.append(filepath)
                        except Exception:
                            continue
            except Exception as e:
                logger.debug(f"Ошибка при поиске пути RTSTRUCT: {e}")
                
            if self.rtstruct_files:
                try:
                    self.rtstruct_files.sort(key=os.path.getmtime)
                except Exception as se:
                    logger.debug(f"Ошибка при сортировке RTSTRUCT файлов: {se}")
                self.existing_rtstruct_path = self.rtstruct_files[-1]
                
                if hasattr(self, 'rtstruct_combo'):
                    self.rtstruct_combo.blockSignals(True)
                    for f in self.rtstruct_files:
                        self.rtstruct_combo.addItem(os.path.basename(f), f)
                    self.rtstruct_combo.setCurrentIndex(len(self.rtstruct_files) - 1)
                    self.rtstruct_combo.setEnabled(True)
                    self.chk_show_structures.setEnabled(True)
                    self.rtstruct_combo.blockSignals(False)
                    
                if hasattr(self, 'merge_rtstruct_combo'):
                    self.merge_rtstruct_combo.blockSignals(True)
                    self.merge_rtstruct_combo.clear()
                    for f in self.rtstruct_files:
                        self.merge_rtstruct_combo.addItem(os.path.basename(f), f)
                    self.merge_rtstruct_combo.setCurrentIndex(len(self.rtstruct_files) - 1)
                    self.merge_rtstruct_combo.blockSignals(False)
                
                self.update_merge_combo_state()

                if self.chk_show_structures.isChecked():
                    self.on_show_structures_changed()
                else:
                    self.update_organs_list_highlighting()

        def update_merge_combo_state(self):
            """Обновляет доступность выпадающего списка RTSTRUCT в настройках."""
            if hasattr(self, 'merge_rtstruct_combo'):
                is_merge = self.radio_merge_merge.isChecked()
                has_files = self.merge_rtstruct_combo.count() > 0
                self.merge_rtstruct_combo.setEnabled(is_merge and has_files)

        def on_viewer_rtstruct_changed(self, index: int):
            """Синхронизирует выбор RTSTRUCT во вьюере с комбобоксом в настройках слияния."""
            if index < 0:
                return
            path = self.rtstruct_combo.itemData(index)
            self.existing_rtstruct_path = path
            
            if hasattr(self, 'merge_rtstruct_combo'):
                self.merge_rtstruct_combo.blockSignals(True)
                self.merge_rtstruct_combo.setCurrentIndex(index)
                self.merge_rtstruct_combo.blockSignals(False)
                
            self.on_show_structures_changed()

        def on_merge_rtstruct_changed(self, index: int):
            """Синхронизирует выбор RTSTRUCT в настройках с комбобоксом во вьюере."""
            if index < 0:
                return
            path = self.merge_rtstruct_combo.itemData(index)
            self.existing_rtstruct_path = path
            
            if hasattr(self, 'rtstruct_combo'):
                self.rtstruct_combo.blockSignals(True)
                self.rtstruct_combo.setCurrentIndex(index)
                self.rtstruct_combo.blockSignals(False)
                
            if self.chk_show_structures.isChecked():
                self.on_show_structures_changed()

        def _clear_roi_overlay(self, permanent: bool = False):
            if hasattr(self, 'roi_overlay_item') and self.roi_overlay_item is not None:
                if permanent:
                    try:
                        if self.roi_overlay_item in self.dicom_viewer.getView().addedItems:
                            self.dicom_viewer.getView().removeItem(self.roi_overlay_item)
                    except Exception:
                        pass
                    if hasattr(self, 'roi_overlay_item'):
                        del self.roi_overlay_item
                else:
                    self.roi_overlay_item.setVisible(False)
            if hasattr(self, 'roi_overlay_3d'):
                del self.roi_overlay_3d

        def on_show_structures_toggled(self, state: int):
            if state == 2:  # Qt.CheckState.Checked
                # Предупреждение при просмотре во время сегментации на CPU
                if (hasattr(self, 'worker') and self.worker and self.worker.isRunning()
                        and hasattr(self, 'radio_cpu') and self.radio_cpu.isChecked()):
                    from PyQt6.QtWidgets import QMessageBox
                    msg = QMessageBox(self)
                    msg.setWindowTitle("⚠ Предупреждение — CPU режим")
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setText(
                        "Сейчас идёт автооконтуривание на CPU.\n\n"
                        "Просмотр 3D-структур создаст дополнительную нагрузку на процессор, "
                        "что значительно замедлит и сегментацию, и отрисовку контуров.\n\n"
                        "Рекомендуется дождаться завершения автооконтуривания."
                    )
                    btn_continue = msg.addButton("Всё равно открыть", QMessageBox.ButtonRole.AcceptRole)
                    btn_cancel = msg.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
                    msg.setDefaultButton(btn_cancel)
                    msg.setStyleSheet(self.styleSheet())
                    msg.exec()
                    if msg.clickedButton() == btn_cancel:
                        self.chk_show_structures.blockSignals(True)
                        self.chk_show_structures.setChecked(False)
                        self.chk_show_structures.blockSignals(False)
                        return

                if hasattr(self, 'color_preset_combo') and self.color_preset_combo.currentText() != "Яркий неоновый":
                    self.is_switching_color_preset = True
                    try:
                        self.color_preset_combo.blockSignals(True)
                        self.color_preset_combo.setCurrentText("Яркий неоновый")
                        self.color_preset_combo.blockSignals(False)
                        self.on_color_preset_changed("Яркий неоновый")
                    finally:
                        self.is_switching_color_preset = False
            self.on_show_structures_changed()

        def on_show_structures_changed(self):
            import pyqtgraph as pg
            import numpy as np
            from PyQt6.QtWidgets import QApplication, QProgressDialog
            from PyQt6.QtCore import Qt
            from PyQt6.QtGui import QBrush, QColor, QPixmap, QIcon
            import re
            
            if not getattr(self, 'current_dicom_dir', None) or getattr(self, 'volume_3d_base', None) is None:
                self._clear_roi_overlay(permanent=False)
                self._clear_imported_organs()
                self._last_loaded_rtstruct = None
                self.update_organs_list_highlighting()
                self.update_run_button(bool(self.series_table.selectedItems()))
                return
                
            if not getattr(self, 'chk_show_structures', None) or not self.chk_show_structures.isChecked():
                self._clear_roi_overlay(permanent=False)
                self._clear_imported_organs()
                self._last_loaded_rtstruct = None
                
                # Возвращаем цветовую гамму QUANTEC при выходе из просмотра
                if hasattr(self, 'color_preset_combo') and self.color_preset_combo.currentText() != "Цвета QUANTEC":
                    self.is_switching_color_preset = True
                    try:
                        self.color_preset_combo.blockSignals(True)
                        self.color_preset_combo.setCurrentText("Цвета QUANTEC")
                        self.color_preset_combo.blockSignals(False)
                        self.on_color_preset_changed("Цвета QUANTEC")
                    finally:
                        self.is_switching_color_preset = False
                
                # Делаем доступными выбор пресетов и кнопки "выбрать все"/"снять все"
                if hasattr(self, 'preset_combo'):
                    self.preset_combo.setEnabled(True)
                if hasattr(self, 'btn_select_all'):
                    self.btn_select_all.setEnabled(True)
                if hasattr(self, 'btn_deselect_all'):
                    self.btn_deselect_all.setEnabled(True)
                
                # Возвращаем видимость всем стандартным органам
                self.organs_list.blockSignals(True)
                for i in range(self.organs_list.count()):
                    item = self.organs_list.item(i)
                    item.setHidden(False)
                self.organs_list.blockSignals(False)
                
                # Возвращаем левую панель и сплиттеры к стандартным размерам
                if hasattr(self, 'left_card') and hasattr(self, 'splitter'):
                    self.left_card.setMinimumWidth(490)
                    self.left_card.setMaximumWidth(max(490, int(self.width() * 0.5)))
                    self.splitter.setSizes([490, 430])
                if hasattr(self, 'main_splitter'):
                    self.main_splitter.setSizes([600, 400])
                if hasattr(self, 'v_splitter'):
                    self.v_splitter.setSizes([500, 500])
                if hasattr(self, 'dicom_viewer'):
                    try:
                        self.dicom_viewer.getView().autoRange()
                    except Exception:
                        pass
                
                self.update_organs_list_highlighting()
                self.update_run_button(bool(self.series_table.selectedItems()))
                return
                


            # Делаем недоступными выбор пресетов и кнопки "выбрать все"/"снять все"
            if hasattr(self, 'preset_combo'):
                self.preset_combo.setEnabled(False)
            if hasattr(self, 'btn_select_all'):
                self.btn_select_all.setEnabled(False)
            if hasattr(self, 'btn_deselect_all'):
                self.btn_deselect_all.setEnabled(False)
                
            rtstruct_path = self.rtstruct_combo.currentData()
            if not rtstruct_path or not os.path.exists(rtstruct_path):
                self._clear_roi_overlay(permanent=False)
                self._clear_imported_organs()
                self._last_loaded_rtstruct = None
                
                # Возвращаем левую панель и сплиттеры к стандартным размерам при отсутствии файла
                if hasattr(self, 'left_card') and hasattr(self, 'splitter'):
                    self.left_card.setMinimumWidth(490)
                    self.left_card.setMaximumWidth(max(490, int(self.width() * 0.5)))
                    self.splitter.setSizes([490, 430])
                if hasattr(self, 'main_splitter'):
                    self.main_splitter.setSizes([600, 400])
                if hasattr(self, 'v_splitter'):
                    self.v_splitter.setSizes([500, 500])
                if hasattr(self, 'dicom_viewer'):
                    try:
                        self.dicom_viewer.getView().autoRange()
                    except Exception:
                        pass
                
                self.update_organs_list_highlighting()
                self.update_run_button(bool(self.series_table.selectedItems()))
                return
                
            # Увеличиваем вьюер за счет сжатия таблицы КТ и логов (левая панель со структурами сохраняет стандартный размер!)
            if hasattr(self, 'left_card') and hasattr(self, 'splitter'):
                self.left_card.setMinimumWidth(490)
                self.left_card.setMaximumWidth(max(490, int(self.width() * 0.5)))
                self.splitter.setSizes([490, 430])
            if hasattr(self, 'main_splitter'):
                self.main_splitter.setSizes([150, 850])
            if hasattr(self, 'v_splitter'):
                self.v_splitter.setSizes([850, 150])
            if hasattr(self, 'dicom_viewer'):
                try:
                    self.dicom_viewer.getView().autoRange()
                except Exception:
                    pass
            
            is_new_rtstruct = (getattr(self, "_last_loaded_rtstruct", None) != rtstruct_path)
            
            # Инициализируем кэш масок лениво
            if not hasattr(self, "_loaded_roi_masks") or is_new_rtstruct:
                self._loaded_roi_masks = {}
                self._cached_rtstruct = None
                self._cached_rtstruct_path = None
                
            # Убеждаемся, что roi_overlay_item создан и добавлен во вьюер
            if not hasattr(self, 'roi_overlay_item') or self.roi_overlay_item is None:
                self.roi_overlay_item = pg.ImageItem()
                self.roi_overlay_item.setZValue(10)
                self.dicom_viewer.getView().addItem(self.roi_overlay_item)
            self.roi_overlay_item.setVisible(True)
            
            progress_dialog = None
            try:
                # Всегда меняем курсор на WaitCursor для предупреждения о процессе отрисовки
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.status_step_label.setText("⏳ Отрисовка 3D-структур...")
                QApplication.processEvents()

                # Избегаем повторного тяжелого парсинга RTSTRUCT с диска
                if not getattr(self, "_cached_rtstruct", None) or getattr(self, "_cached_rtstruct_path", None) != rtstruct_path:
                    self.status_step_label.setText("⏳ Подготовка 3D-сцены: чтение DICOM RTSTRUCT файла...")
                    QApplication.processEvents()
                    
                    from rt_utils import RTStructBuilder
                    self._cached_rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=self.current_dicom_dir,
                        rt_struct_path=rtstruct_path,
                        warn_only=True
                    )
                    self._cached_rtstruct_path = rtstruct_path
                
                rtstruct = self._cached_rtstruct
                roi_names = rtstruct.get_roi_names()
                total_rois = len(roi_names)
                
                # Умное нечеткое сопоставление (Fuzzy Matching) OAR
                normalize_name = lambda n: re.sub(r'[^a-z0-9]', '', n.lower())
                all_supported_organs = self.engine.get_all_supported_organs()
                
                # Создаем быстрый маппинг нормализованных имен на их технические ID
                supported_norm_map = {}
                for org in all_supported_organs:
                    supported_norm_map[normalize_name(org)] = org
                    mon_pretty = self.engine.get_monaco_pretty_name(org)
                    if mon_pretty:
                        supported_norm_map[normalize_name(mon_pretty)] = org
                
                # Дополняем виртуальными органами из ORGAN_GROUPS (lungs и т.д.),
                # которых нет в TotalSegmentator class_map
                for group_organs in ORGAN_GROUPS.values():
                    for org in group_organs:
                        if normalize_name(org) not in supported_norm_map:
                            supported_norm_map[normalize_name(org)] = org
                            mon_pretty = self.engine.get_monaco_pretty_name(org)
                            if mon_pretty:
                                supported_norm_map[normalize_name(mon_pretty)] = org

                def get_mapped_organ(roi_name_str: str) -> str:
                    norm = normalize_name(roi_name_str)
                    if norm in EXTERNAL_ALIASES:
                        return EXTERNAL_ALIASES[norm]
                    if norm in supported_norm_map:
                        return supported_norm_map[norm]
                    return roi_name_str.lower().replace(" ", "_")

                # Извлекаем оригинальные цвета ROI из DICOM RTSTRUCT
                roi_colors = {}
                try:
                    if hasattr(rtstruct, 'ds'):
                        roi_number_to_name = {}
                        for roi_set in getattr(rtstruct.ds, 'StructureSetROISequence', []):
                            roi_number_to_name[int(roi_set.ROINumber)] = roi_set.ROIName
                            
                        for roi_contour in getattr(rtstruct.ds, 'ROIContourSequence', []):
                            ref_num = int(getattr(roi_contour, 'ReferencedROINumber', -1))
                            if ref_num in roi_number_to_name:
                                roi_name = roi_number_to_name[ref_num]
                                if hasattr(roi_contour, 'ROIDisplayColor'):
                                    roi_colors[roi_name] = [int(c) for c in roi_contour.ROIDisplayColor]
                except Exception as ex:
                    logger.warning(f"Не удалось прочитать оригинальные цвета ROI: {ex}")

                # Собираем перечень стандартных OAR
                gui_supported_organs = set()
                for i in range(self.organs_list.count()):
                    itm = self.organs_list.item(i)
                    itm_data = itm.data(Qt.ItemDataRole.UserRole)
                    if itm_data != "header" and itm_data and itm.data(Qt.ItemDataRole.UserRole + 1) is not True:
                        org_name = itm_data
                        if org_name:
                            gui_supported_organs.add(org_name.lower())

                # Распределяем ROI на стандартные (присутствующие в файле) и сторонние (для импорта)
                file_supported_organs = set()
                imported_rois = []
                for roi in roi_names:
                    orig_organ = get_mapped_organ(roi)
                    orig_organ_lower = orig_organ.lower()
                    if orig_organ_lower in gui_supported_organs:
                        file_supported_organs.add(orig_organ_lower)
                    else:
                        imported_rois.append((roi, orig_organ))

                # Пересоздаем динамические импортированные строки только при первой загрузке нового RTSTRUCT
                if is_new_rtstruct:
                    self._clear_imported_organs()
                    if imported_rois:
                        self.organs_list.blockSignals(True)
                        try:
                            header_item = QListWidgetItem(f"━━━ ИМПОРТИРОВАННЫЕ OAR ━━━ ({len(imported_rois)})")
                            header_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                            header_item.setCheckState(Qt.CheckState.Unchecked)
                            header_item.setData(Qt.ItemDataRole.UserRole, "header")
                            header_item.setData(Qt.ItemDataRole.UserRole + 1, True)
                            
                            font = header_item.font()
                            font.setBold(True)
                            header_item.setFont(font)
                            header_item.setForeground(QBrush(QColor("#007acc")))
                            header_item.setBackground(QBrush(QColor("#242424")))
                            
                            self.organs_list.addItem(header_item)
                            self.imported_items.append(header_item)
                            
                            for roi, orig_organ in imported_rois:
                                color_rgb = roi_colors.get(roi, [0, 255, 128])
                                # Кэшируем цвет стороннего органа в движке
                                self.engine.colors[orig_organ] = color_rgb
                                
                                ru_name = self.engine.ru_names.get(orig_organ, roi)
                                item = QListWidgetItem(f"   {ru_name}")
                                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                                item.setCheckState(Qt.CheckState.Checked)  # По умолчанию включен
                                item.setData(Qt.ItemDataRole.UserRole, orig_organ)
                                item.setData(Qt.ItemDataRole.UserRole + 1, True)
                                
                                # Устанавливаем цветную иконку
                                pixmap = QPixmap(14, 14)
                                pixmap.fill(QColor(color_rgb[0], color_rgb[1], color_rgb[2]))
                                item.setIcon(QIcon(pixmap))
                                
                                self.organs_list.addItem(item)
                                self.imported_items.append(item)
                        finally:
                            self.organs_list.blockSignals(False)

                file_organs = set(get_mapped_organ(r) for r in roi_names)

                # Если это первая загрузка этого файла RTSTRUCT, интеллектуально отмечаем только те органы, которые в нем есть
                if is_new_rtstruct:
                    self.organs_list.blockSignals(True)
                    self.is_updating_presets = True
                    for i in range(self.organs_list.count()):
                        itm = self.organs_list.item(i)
                        itm_data = itm.data(Qt.ItemDataRole.UserRole)
                        if itm_data != "header":
                            if isinstance(itm_data, dict):
                                org_name = itm_data.get("name") or (list(itm_data.keys())[0] if itm_data else "")
                            else:
                                org_name = itm_data
                            
                            if org_name and get_mapped_organ(org_name) in file_organs:
                                itm.setCheckState(Qt.CheckState.Checked)
                            else:
                                itm.setCheckState(Qt.CheckState.Unchecked)
                    
                    self.update_headers_check_states()
                    self._sync_preset_combo_to_organs()
                    self.organs_list.blockSignals(False)
                    self.is_updating_presets = False
                    self.update_checked_organs_count()
                    self._last_loaded_rtstruct = rtstruct_path

                # Управляем видимостью стандартных органов и пустых заголовков групп
                self.organs_list.blockSignals(True)
                try:
                    group_structures = []
                    last_header = None
                    current_group_items = []
                    
                    for i in range(self.organs_list.count()):
                        item = self.organs_list.item(i)
                        itm_data = item.data(Qt.ItemDataRole.UserRole)
                        if itm_data == "header":
                            if last_header is not None:
                                group_structures.append((last_header, current_group_items))
                            last_header = item
                            current_group_items = []
                        elif last_header is not None:
                            current_group_items.append(item)
                            
                    if last_header is not None:
                        group_structures.append((last_header, current_group_items))
                            
                    for header, items in group_structures:
                        group_visible = False
                        for item in items:
                            if item.data(Qt.ItemDataRole.UserRole + 1) is True:
                                item.setHidden(False)
                                group_visible = True
                            else:
                                itm_data = item.data(Qt.ItemDataRole.UserRole)
                                orig_organ_lower = itm_data.lower() if isinstance(itm_data, str) else ""
                                if orig_organ_lower in file_supported_organs:
                                    item.setHidden(False)
                                    group_visible = True
                                else:
                                    item.setHidden(True)
                        
                        header.setHidden(not group_visible)
                except Exception as filter_e:
                    logger.error(f"Ошибка при фильтрации списка органов по видимости: {filter_e}")
                finally:
                    self.organs_list.blockSignals(False)

                # Собираем чекбоксы, которые сняты (для фильтрации вывода оверлея)
                unchecked_organs = set()
                for i in range(self.organs_list.count()):
                    itm = self.organs_list.item(i)
                    itm_data = itm.data(Qt.ItemDataRole.UserRole)
                    if itm_data != "header" and itm_data:
                        if itm.checkState() == Qt.CheckState.Unchecked:
                            unchecked_organs.add(itm_data.lower() if isinstance(itm_data, str) else "")
                
                # Фильтруем список структур, которые реально будем отрисовывать
                rois_to_draw = []
                for roi in roi_names:
                    orig_organ = get_mapped_organ(roi)
                    orig_organ_lower = orig_organ.lower()
                    
                    # Орган должен отображаться, если его чекбокс НЕ снят в списке OAR

                    if orig_organ_lower not in unchecked_organs:
                        rois_to_draw.append((roi, orig_organ))

                # Проверяем, все ли нужные маски уже есть в кэше оперативной памяти
                all_cached = all(roi in self._loaded_roi_masks for roi, _ in rois_to_draw)

                if not all_cached:
                    # Создаем красивое модальное окно прогресса при первом чтении масок
                    progress_dialog = QProgressDialog("⏳ Инициализация 3D-структур...", None, 0, len(rois_to_draw), self)
                    progress_dialog.setWindowTitle("Загрузка 3D-контуров")
                    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                    progress_dialog.setMinimumDuration(0)
                    progress_dialog.setCancelButton(None)
                    progress_dialog.setStyleSheet(self.styleSheet())
                    progress_dialog.setValue(0)
                    progress_dialog.show()
                    QApplication.processEvents()

                z_dim, x_dim, y_dim = self.volume_3d_base.shape
                overlay_3d = np.zeros((z_dim, x_dim, y_dim, 4), dtype=np.uint8)
                
                for idx, (roi, orig_organ) in enumerate(rois_to_draw, start=1):
                    try:
                        ru_name = self.engine.ru_names.get(orig_organ, orig_organ)
                        if orig_organ == "body":
                            ru_name = "Контур тела (Body)"
                        
                        # Обновляем прогресс, только если идет реальный расчет
                        if not all_cached:
                            if progress_dialog:
                                progress_dialog.setValue(idx)
                                progress_dialog.setLabelText(f"⏳ Отрисовка контуров: {ru_name} ({idx}/{len(rois_to_draw)})...")
                            self.status_step_label.setText(f"⏳ Отрисовка контуров: {ru_name} ({idx}/{len(rois_to_draw)})...")
                            QApplication.processEvents()
                        
                        # Извлекаем маску: либо из кэша (мгновенно), либо рассчитываем с диска (один раз)
                        if roi in self._loaded_roi_masks:
                            mask_3d = self._loaded_roi_masks[roi]
                        else:
                            # Безопасный slice-by-slice экстрактор маски
                            mask_3d = self._get_roi_mask_safe(
                                rtstruct, roi,
                                self.volume_3d_base.shape,
                                getattr(self, 'z_positions', None),
                                getattr(self, 'dicom_pixel_spacing', (1.0, 1.0)),
                                getattr(self, 'dicom_image_position', [0.0, 0.0])
                            )
                            self._loaded_roi_masks[roi] = mask_3d
                        
                        if orig_organ == "body":
                            if "body_boundary" in self._loaded_roi_masks:
                                boundary = self._loaded_roi_masks["body_boundary"]
                            else:
                                import cv2
                                kernel = np.ones((5, 5), dtype=np.uint8)
                                eroded_slices = []
                                for z in range(mask_3d.shape[0]):
                                    slice_2d = mask_3d[z].astype(np.uint8)
                                    eroded_slice = cv2.erode(slice_2d, kernel, iterations=1)
                                    eroded_slices.append(eroded_slice.astype(bool))
                                boundary = mask_3d ^ np.stack(eroded_slices)
                                self._loaded_roi_masks["body_boundary"] = boundary
                                
                            color = self.engine.colors.get(orig_organ, [0, 255, 0])
                            overlay_3d[boundary, 0] = color[0]
                            overlay_3d[boundary, 1] = color[1]
                            overlay_3d[boundary, 2] = color[2]
                            overlay_3d[boundary, 3] = 255
                        else:
                            color = self.engine.colors.get(orig_organ, [0, 255, 128])
                            overlay_3d[mask_3d, 0] = color[0]
                            overlay_3d[mask_3d, 1] = color[1]
                            overlay_3d[mask_3d, 2] = color[2]
                            overlay_3d[mask_3d, 3] = 100
                    except Exception as roi_e:
                        logger.warning(f"Не удалось отрисовать структуру {roi}: {roi_e}")
                
                # Бесшовно обновляем 3D-данные оверлея
                self.roi_overlay_3d = overlay_3d
                self.update_roi_overlay_frame()
                self.status_step_label.setText("Текущий шаг: Ожидание запуска...")
                
                # Обновляем кнопку автооконтуривания (блокируем в режиме просмотра)
                self.update_run_button(bool(self.series_table.selectedItems()))
                # Обновляем подсветку списка органов
                self.update_organs_list_highlighting()
            except Exception as e:
                logger.error(f"Ошибка загрузки структур во вьюер: {e}")
                self.status_step_label.setText("Текущий шаг: Ожидание запуска...")
                self.update_run_button(bool(self.series_table.selectedItems()))
                self.update_organs_list_highlighting()
            finally:
                if progress_dialog:
                    progress_dialog.close()
                QApplication.restoreOverrideCursor()

        def _get_roi_mask_safe(
            self,
            rtstruct,
            roi_name: str,
            volume_shape: tuple,
            z_positions: list | None,
            pixel_spacing: tuple = (1.0, 1.0),
            image_position: list = (0.0, 0.0)
        ):
            """
            Безопасный slice-by-slice экстрактор маски ROI из DICOM RTSTRUCT.

            В отличие от rt_utils.get_roi_mask_by_name(), не падает при
            вырожденных контурах (< 3 точек), характерных для Skull, Eye и др.
            Возвращает numpy bool-маску формата (z, x, y), совместимую с overlay_3d.
            """
            import numpy as np
            import cv2

            z_dim, x_dim, y_dim = volume_shape
            mask = np.zeros((z_dim, x_dim, y_dim), dtype=bool)

            # Ищем ROI по имени в DICOM-датасете
            try:
                roi_index = None
                for i, roi_item in enumerate(rtstruct.ds.StructureSetROISequence):
                    if roi_item.ROIName == roi_name:
                        roi_index = i
                        break
                if roi_index is None:
                    return mask

                # Находим соответствующий элемент в ROIContourSequence
                roi_contour = None
                for rc in rtstruct.ds.ROIContourSequence:
                    if int(getattr(rc, 'ReferencedROINumber', -1)) == int(rtstruct.ds.StructureSetROISequence[roi_index].ROINumber):
                        roi_contour = rc
                        break
                if roi_contour is None or not hasattr(roi_contour, 'ContourSequence'):
                    return mask

            except Exception:
                return mask

            # Строим маппинг Z-координата -> индекс среза
            z_index_map: dict[float, int] = {}
            if z_positions:
                for idx, z in enumerate(z_positions):
                    z_index_map[round(z, 2)] = idx

            ps_row, ps_col = pixel_spacing  # мм/пиксель по строкам и столбцам
            img_pos_x, img_pos_y = image_position  # позиция первого пикселя (мм)

            for contour in roi_contour.ContourSequence:
                try:
                    pts_raw = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
                    n_pts = len(pts_raw)

                    # Пропускаем вырожденные контуры — именно они вызывают cv2.fillPoly assert
                    if n_pts < 3:
                        continue

                    z_val = round(float(pts_raw[0, 2]), 2)

                    # Находим ближайший Z-индекс
                    if z_val in z_index_map:
                        z_idx = z_index_map[z_val]
                    elif z_positions:
                        # Ближайший срез (допуск ±2 мм)
                        diffs = [abs(z - z_val) for z in z_positions]
                        min_diff = min(diffs)
                        if min_diff > 2.0:
                            continue
                        z_idx = diffs.index(min_diff)
                    else:
                        continue

                    # Конвертируем мировые координаты (мм) -> пиксельные (с учётом разрешения)
                    x_pts_mm = pts_raw[:, 0]  # X в мировых координатах
                    y_pts_mm = pts_raw[:, 1]  # Y в мировых координатах

                    # Формула: pixel = (world - image_position) / pixel_spacing
                    col_pts = np.round((x_pts_mm - img_pos_x) / ps_col).astype(np.int32)
                    row_pts = np.round((y_pts_mm - img_pos_y) / ps_row).astype(np.int32)

                    # Зажимаем координаты в границах среза
                    col_pts = np.clip(col_pts, 0, x_dim - 1)
                    row_pts = np.clip(row_pts, 0, y_dim - 1)

                    # Формируем массив точек для cv2.fillPoly
                    contour_pts = np.stack([col_pts, row_pts], axis=1).reshape((-1, 1, 2))

                    if len(contour_pts) < 3:
                        continue

                    # Рисуем на пустом срезе
                    slice_mask = np.zeros((y_dim, x_dim), dtype=np.uint8)
                    cv2.fillPoly(slice_mask, [contour_pts.astype(np.int32)], 1)
                    mask[z_idx] |= slice_mask.T.astype(bool)

                except Exception as ce:
                    logger.debug(f"[{roi_name}] Пропуск контура (вырожденный): {ce}")
                    continue

            return mask

        def update_roi_overlay_frame(self):
            if hasattr(self, 'roi_overlay_item') and hasattr(self, 'roi_overlay_3d') and self.roi_overlay_item is not None:
                if self.roi_overlay_item.isVisible() and self.roi_overlay_3d is not None:
                    idx = self.dicom_viewer.currentIndex
                    if idx < self.roi_overlay_3d.shape[0]:
                        self.roi_overlay_item.setImage(self.roi_overlay_3d[idx], autoLevels=False)

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
            
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Все органы (All)")
            self.preset_combo.blockSignals(False)
            
            self.update_headers_check_states()
            self.save_settings()
            self.update_checked_organs_count()
            
            if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                self.on_show_structures_changed()

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
            
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Пользовательский (Custom)")
            self.preset_combo.blockSignals(False)
            
            self.update_headers_check_states()
            self.save_settings()
            self.update_checked_organs_count()
            
            if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                self.on_show_structures_changed()

        def _sync_preset_combo_to_organs(self):
            """Подбирает и устанавливает в комбобоксе пресет, соответствующий текущим выбранным органам.
            Если точного совпадения нет — оставляет заглушку «— Выберите пресет —»."""
            checked_organs = []
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                org = item.data(Qt.ItemDataRole.UserRole)
                if org != "header" and item.checkState() == Qt.CheckState.Checked:
                    if org not in checked_organs:
                        checked_organs.append(org)

            matched = "— Выберите пресет —"
            if checked_organs:
                all_orgs = list(self.engine.ru_names.keys())
                if set(checked_organs) == set(all_orgs):
                    matched = "Все органы (All)"
                else:
                    for pname, porgans in self.engine.presets.items():
                        porgans_flat = []
                        for po in porgans:
                            if isinstance(po, dict):
                                porgans_flat.extend(po.keys())
                            else:
                                porgans_flat.append(po)
                        if set(checked_organs) == set(porgans_flat):
                            matched = pname
                            break
                    else:
                        matched = "Пользовательский (Custom)"

            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(matched)
            self.preset_combo.blockSignals(False)

        def apply_preset_checked_states(self, preset_name: str):
            """Снимает/ставит галочки в списке в соответствии с выбранным пресетом."""
            if preset_name == "Пользовательский (Custom)":
                return

            if preset_name in self.engine.presets:
                target_organs_raw = self.engine.presets.get(preset_name, [])
                target_organs = []
                for item in target_organs_raw:
                    if isinstance(item, dict):
                        target_organs.extend(item.keys())
                    else:
                        target_organs.append(item)
            elif preset_name in ORGAN_GROUPS:
                target_organs = ORGAN_GROUPS[preset_name]
            elif preset_name == "Все органы (All)":
                target_organs = self.engine.get_all_supported_organs()
            else:
                target_organs = []

            # Блокируем сигналы чтобы не вызывать on_organ_item_changed в цикле
            self.organs_list.blockSignals(True)
            try:
                for i in range(self.organs_list.count()):
                    item = self.organs_list.item(i)
                    organ_name = item.data(Qt.ItemDataRole.UserRole)
                    if organ_name == "header":
                        continue
                    # Игнорируем импортированные органы при смене пресета OAR
                    if item.data(Qt.ItemDataRole.UserRole + 1) is True:
                        continue
                    if organ_name in target_organs:
                        # Проверяем, не заблокирован ли орган из-за отсутствия лицензии
                        task = ROI_TO_TASK_MAP.get(organ_name, 'total')
                        is_licensed_task = task in LICENSED_TASKS
                        has_license = hasattr(self.engine, "licenses") and isinstance(self.engine.licenses, str) and self.engine.licenses.strip()
                        
                        if is_licensed_task and not has_license:
                            # Заблокированный орган не чекаем пресетом
                            item.setCheckState(Qt.CheckState.Unchecked)
                        else:
                            item.setCheckState(Qt.CheckState.Checked)
                    else:
                        item.setCheckState(Qt.CheckState.Unchecked)
            finally:
                self.organs_list.blockSignals(False)

            # После обновления органов — пересчитываем состояния заголовков групп
            self.update_headers_check_states()
            self.update_checked_organs_count()

        def update_headers_check_states(self):
            """Обновляет состояния чекбоксов заголовков на основе состояния дочерних органов."""
            self.is_updating_presets = True
            current_header = None
            group_items = []

            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                role = item.data(Qt.ItemDataRole.UserRole)
                if role == "header":
                    if current_header is not None:
                        self._set_header_state_from_children(current_header, group_items)
                    current_header = item
                    group_items = []
                else:
                    group_items.append(item)

            if current_header is not None:
                self._set_header_state_from_children(current_header, group_items)
            self.is_updating_presets = False

        def _set_header_state_from_children(self, header_item: QListWidgetItem, children: list):
            if not children:
                header_item.setCheckState(Qt.CheckState.Unchecked)
                return
            checked_count = sum(1 for item in children if item.checkState() == Qt.CheckState.Checked)
            if checked_count == len(children):
                header_item.setCheckState(Qt.CheckState.Checked)
            elif checked_count == 0:
                header_item.setCheckState(Qt.CheckState.Unchecked)
            else:
                header_item.setCheckState(Qt.CheckState.PartiallyChecked)

        def update_checked_organs_count(self):
            """Подсчитывает отмеченные органы и обновляет надпись organs_header."""
            if not hasattr(self, 'organs_header'):
                return
            total = 0
            checked = 0
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if data and data != "header":
                    total += 1
                    if item.checkState() == Qt.CheckState.Checked:
                        checked += 1
            self.organs_header.setText(f"Органы для автооконтурирования: {checked} из {total}")

        def on_preset_changed(self, index: int):
            """Слот изменения выбранного пресета (вызывается при каждом выборе из списка)."""
            text = self.preset_combo.itemText(index)
            if text == "— Выберите пресет —":
                return  # Заглушка — ничего не делаем
            self.is_updating_presets = True
            self.apply_preset_checked_states(text)
            self.is_updating_presets = False
            self.save_settings()
            
            if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                self.on_show_structures_changed()

        def on_organ_item_changed(self, item: QListWidgetItem):
            """Слот изменения состояния чекбокса органа пользователем."""
            if self.is_updating_presets:
                return

            try:
                self.organs_list.blockSignals(True)
                
                data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    organ_name = data.get("name") or list(data.keys())[0] if data else ""
                else:
                    organ_name = data
                
                if not organ_name:
                    return
                    
                if organ_name == "header":
                    # Это клик по заголовку группы!
                    self.is_updating_presets = True
                    new_state = item.checkState()
                    if new_state == Qt.CheckState.PartiallyChecked:
                        new_state = Qt.CheckState.Checked
                        item.setCheckState(new_state)

                    # Находим все органы в этой группе и ставим им новое состояние
                    row = self.organs_list.row(item)
                    changed_organs = []
                    for i in range(row + 1, self.organs_list.count()):
                        next_item = self.organs_list.item(i)
                        next_role = next_item.data(Qt.ItemDataRole.UserRole)
                        if next_role == "header":
                            break
                        next_item.setCheckState(new_state)
                        changed_organs.append(next_role)
                    self.is_updating_presets = False

                    # Синхронизируем дубли измененных органов в других группах
                    self.is_updating_presets = True
                    for i in range(self.organs_list.count()):
                        itm = self.organs_list.item(i)
                        itm_role = itm.data(Qt.ItemDataRole.UserRole)
                        if itm_role != "header" and itm_role in changed_organs:
                            itm.setCheckState(new_state)
                    self.is_updating_presets = False
                else:
                    # Это клик по обычному органу
                    self.is_updating_presets = True
                    state = item.checkState()
                    # Синхронизация дубликатов
                    for i in range(self.organs_list.count()):
                        itm = self.organs_list.item(i)
                        if itm != item:
                            itm_data = itm.data(Qt.ItemDataRole.UserRole)
                            if isinstance(itm_data, dict):
                                itm_org = itm_data.get("name") or list(itm_data.keys())[0] if itm_data else ""
                            else:
                                itm_org = itm_data
                            if itm_org == organ_name:
                                itm.setCheckState(state)
                    self.is_updating_presets = False

                # Обновляем состояния всех заголовков групп
                self.update_headers_check_states()
                # Синхронизируем текст в комбобоксе пресетов
                self._sync_preset_combo_to_organs()
                self.save_settings()
                
                # Если включен показ структур, перерисуем их
                if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                    self.on_show_structures_changed()
                
                self.update_checked_organs_count()
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Ошибка выбора", f"Сбой при обработке клика: {e}")
            finally:
                self.organs_list.blockSignals(False)

        def on_organ_item_clicked(self, item: QListWidgetItem):
            """Слот для клика по элементу списка органов."""
            role = item.data(Qt.ItemDataRole.UserRole)
            if role == "header":
                self.toggle_group_collapse(item)

        def toggle_group_collapse(self, header_item: QListWidgetItem):
            """Сворачивает или разворачивает группу органов."""
            role = header_item.data(Qt.ItemDataRole.UserRole)
            if role != "header":
                return
            
            text = header_item.text()
            # Определяем текущее состояние по знаку [+] или [-]
            is_collapsed = text.startswith("[+]")
            new_collapsed = not is_collapsed
            
            new_prefix = "[+] " if new_collapsed else "[-] "
            clean_text = text
            if text.startswith("[+] ") or text.startswith("[-] "):
                clean_text = text[4:]
            
            header_item.setText(f"{new_prefix}{clean_text}")
            
            # Сохраняем состояние в self.collapsed_groups
            group_key = clean_text.split(" (")[0]
            if "ОСТАЛЬНОЕ" in group_key:
                self.collapsed_groups["Остальное"] = new_collapsed
            else:
                self.collapsed_groups[group_key] = new_collapsed
            
            # Скрываем/показываем элементы
            self.organs_list.blockSignals(True)
            row = self.organs_list.row(header_item)
            for i in range(row + 1, self.organs_list.count()):
                next_item = self.organs_list.item(i)
                next_role = next_item.data(Qt.ItemDataRole.UserRole)
                if next_role == "header":
                    break
                next_item.setHidden(new_collapsed)
            self.organs_list.blockSignals(False)

        def on_organ_selection_changed(self):
            pass

        def pick_organ_color(self):
            """Открывает диалог выбора цвета для выделенного органа."""
            selected = self.organs_list.selectedItems()
            if not selected:
                return
            
            item = selected[0]
            organ_name = item.data(Qt.ItemDataRole.UserRole)
            if organ_name == "header":
                return
            
            current_rgb = self.engine.colors.get(organ_name, [128, 128, 128])
            initial_color = QColor(current_rgb[0], current_rgb[1], current_rgb[2])
            
            ru_name = self.engine.ru_names.get(organ_name, organ_name)
            new_color = QColorDialog.getColor(initial_color, self, f"Выберите цвет для: {ru_name}")
            
            if new_color.isValid():
                new_rgb = [new_color.red(), new_color.green(), new_color.blue()]
                self.engine.colors[organ_name] = new_rgb
                self.engine.save_presets_config()
                
                # Обновляем иконку
                self.update_item_color_icon(item, organ_name)
                
                # Обновляем все вхождения этого органа в списке (если сдублировано)
                for i in range(self.organs_list.count()):
                    itm = self.organs_list.item(i)
                    if itm.data(Qt.ItemDataRole.UserRole) == organ_name:
                        self.update_item_color_icon(itm, organ_name)
                        
                logger.info(f"Цвет органа {organ_name} успешно изменен на {new_rgb}")
                
                # Если включен показ структур, перерисуем их с новым цветом
                if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                    self.on_show_structures_changed()

        def on_color_preset_changed(self, text: str):
            """Слот изменения цветового пресета."""
            # Наборы пресетов
            preset_palettes = {
                "Классический": {"spleen": [156, 39, 176], "kidney_right": [3, 169, 244], "kidney_left": [33, 150, 243], "gallbladder": [76, 175, 80], "liver": [139, 195, 74], "stomach": [255, 152, 0], "aorta": [244, 67, 54], "inferior_vena_cava": [63, 81, 181], "urinary_bladder": [255, 235, 59], "heart": [233, 30, 99], "lung_left": [0, 150, 136], "lung_right": [0, 188, 212], "lungs": [0, 172, 193], "trachea": [121, 85, 72], "esophagus": [158, 158, 158], "pancreas": [255, 193, 7], "duodenum": [173, 20, 87], "adrenal_gland_left": [255, 87, 34], "adrenal_gland_right": [255, 112, 67], "pulmonary_artery": [0, 150, 255], "small_bowel": [103, 58, 183], "prostate": [233, 30, 99], "rectum": [121, 85, 72], "colon": [0, 121, 107], "femur_left": [255, 224, 178], "femur_right": [255, 224, 178], "hip_left": [230, 238, 156], "hip_right": [230, 238, 156], "sacrum": [141, 110, 99], "spinal_cord": [0, 255, 0], "thyroid_gland": [255, 105, 180], "skull": [255, 228, 196], "brain": [135, 206, 250], "common_carotid_artery_left": [220, 20, 60], "common_carotid_artery_right": [220, 20, 60], "superior_vena_cava": [70, 130, 180], "portal_vein_and_splenic_vein": [0, 139, 139], "clavicula_left": [244, 164, 96], "clavicula_right": [244, 164, 96], "sternum": [222, 184, 135], "iliac_artery_left": [255, 99, 71], "iliac_artery_right": [255, 99, 71], "eye_left": [255, 255, 0], "eye_right": [255, 255, 0], "lens_left": [255, 165, 0], "lens_right": [255, 165, 0], "brain_stem": [210, 105, 30], "optic_nerve_left": [240, 230, 140], "optic_nerve_right": [240, 230, 140]},
                "Цвета QUANTEC": {"spleen": [160, 32, 240], "kidney_right": [0, 0, 255], "kidney_left": [30, 144, 255], "gallbladder": [0, 255, 0], "liver": [34, 139, 34], "stomach": [218, 165, 32], "aorta": [55, 197, 94], "inferior_vena_cava": [194, 166, 130], "urinary_bladder": [255, 215, 0], "heart": [255, 0, 0], "lung_left": [86, 123, 174], "lung_right": [195, 54, 110], "lungs": [140, 88, 142], "trachea": [149, 58, 171], "esophagus": [138, 127, 103], "pancreas": [153, 97, 184], "duodenum": [168, 85, 61], "adrenal_gland_left": [114, 125, 152], "adrenal_gland_right": [161, 157, 200], "pulmonary_artery": [98, 122, 139], "small_bowel": [177, 66, 127], "prostate": [152, 133, 118], "rectum": [139, 69, 19], "colon": [191, 68, 120], "femur_left": [135, 139, 183], "femur_right": [159, 155, 157], "hip_left": [146, 175, 165], "hip_right": [85, 193, 174], "sacrum": [96, 111, 190], "spinal_cord": [116, 98, 57], "thyroid_gland": [113, 52, 117], "skull": [94, 188, 72], "brain": [155, 169, 192], "common_carotid_artery_left": [51, 115, 144], "common_carotid_artery_right": [86, 147, 196], "superior_vena_cava": [84, 137, 160], "portal_vein_and_splenic_vein": [113, 127, 112], "clavicula_left": [144, 51, 84], "clavicula_right": [176, 73, 124], "sternum": [85, 68, 152], "iliac_artery_left": [134, 69, 129], "iliac_artery_right": [78, 137, 190], "eye_left": [255, 255, 100], "eye_right": [255, 255, 100], "lens_left": [255, 140, 0], "lens_right": [255, 140, 0], "brain_stem": [139, 69, 19], "optic_nerve_left": [255, 215, 0], "optic_nerve_right": [255, 215, 0]},
                "Яркий неоновый": {"spleen": [255, 0, 255], "kidney_right": [0, 255, 255], "kidney_left": [0, 191, 255], "gallbladder": [50, 205, 50], "liver": [173, 255, 47], "stomach": [255, 165, 0], "aorta": [255, 255, 0], "inferior_vena_cava": [128, 0, 255], "urinary_bladder": [255, 255, 0], "heart": [255, 20, 147], "lung_left": [255, 0, 255], "lung_right": [255, 0, 255], "lungs": [0, 255, 200], "trachea": [128, 255, 0], "esophagus": [0, 128, 255], "pancreas": [0, 128, 255], "duodenum": [255, 255, 0], "adrenal_gland_left": [255, 255, 0], "adrenal_gland_right": [255, 0, 128], "pulmonary_artery": [0, 128, 255], "small_bowel": [0, 0, 255], "prostate": [255, 0, 0], "rectum": [210, 105, 30], "colon": [0, 128, 255], "femur_left": [0, 255, 128], "femur_right": [128, 255, 0], "hip_left": [128, 0, 255], "hip_right": [0, 0, 255], "sacrum": [255, 0, 255], "spinal_cord": [255, 0, 128], "thyroid_gland": [0, 0, 255], "skull": [255, 0, 0], "brain": [0, 0, 255], "common_carotid_artery_left": [0, 255, 0], "common_carotid_artery_right": [0, 0, 255], "superior_vena_cava": [0, 128, 255], "portal_vein_and_splenic_vein": [0, 255, 0], "clavicula_left": [0, 0, 255], "clavicula_right": [255, 0, 128], "sternum": [0, 255, 0], "iliac_artery_left": [128, 0, 255], "iliac_artery_right": [128, 0, 255], "eye_left": [255, 255, 0], "eye_right": [255, 255, 0], "lens_left": [255, 69, 0], "lens_right": [255, 69, 0], "brain_stem": [255, 105, 180], "optic_nerve_left": [255, 215, 0], "optic_nerve_right": [255, 215, 0]}
            }

            palette = preset_palettes.get(text)
            if palette:
                for organ, color in palette.items():
                    self.engine.colors[organ] = color
                
                # Сохраняем в конфигурационные файлы config/
                self.engine.save_presets_config()
                
                # Обновляем все иконки в списке с временной блокировкой сигналов
                self.structures_list.blockSignals(True)
                try:
                    for i in range(self.structures_list.count()):
                        itm = self.structures_list.item(i)
                        org = itm.data(Qt.ItemDataRole.UserRole)
                        if org != "header" and org:
                            # Безопасное получение названия органа
                            if isinstance(org, dict):
                                org_str = org.get("name") or (list(org.keys())[0] if org else "")
                            else:
                                org_str = org
                            self.update_item_color_icon(itm, org_str)
                finally:
                    self.structures_list.blockSignals(False)
                
                # Если включен показ структур, перерисуем их с новыми цветами пресета (только при ручном переключении!)
                if not getattr(self, 'is_switching_color_preset', False):
                    if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked():
                        self.on_show_structures_changed()
                
                logger.info(f"Цветовая гамма переключена на пресет: '{text}'")

        def on_smoothing_check_changed(self, state: int):
            """Слот изменения состояния чекбокса сглаживания."""
            enabled = (state == 2)
            self.smoothing_combo.setEnabled(enabled)

        def append_log(self, message: str, color: str):
            """Потокобезопасное добавление логов в текстовое окно."""
            if not hasattr(self, 'log_edit'):
                if not hasattr(self, '_startup_log_buffer'):
                    self._startup_log_buffer = []
                self._startup_log_buffer.append((message, color))
                return
            import html
            # Умное распознавание pre-formatted HTML-баннеров и разделителей
            if message.startswith("<br>") or message.startswith("<span") or "style=" in message:
                self.log_edit.append(message)
            else:
                safe_message = html.escape(message)
                self.log_edit.append(f"<span style='color: {color};'>{safe_message}</span>")
            self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

        def start_segmentation(self):
            """Запускает процесс сегментации или отменяет его."""
            if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
                reply = QMessageBox.question(
                    self, 
                    "Подтверждение отмены", 
                    "Вы действительно хотите прервать процесс автоматического оконтурирования?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.status_step_label.setText("Отмена процесса...")
                    self.status_step_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                    self.worker.cancel()
                    self.scan_timer.start(15000)
                return

            if getattr(self, 'chk_show_structures', None) and self.chk_show_structures.isChecked():
                self.chk_show_structures.setChecked(False)
                return

            selected = self.series_table.selectedItems()
            if not selected:
                QMessageBox.critical(self, "Ошибка", "Выберите пациента из таблицы для начала оконтурирования.")
                return
                
            row = selected[0].row()
            dicom_dir = self.series_table.item(row, 6).text()

            if not dicom_dir or not os.path.isdir(dicom_dir):
                QMessageBox.critical(self, "Ошибка", "Путь к DICOM-серии недействителен!")
                return
                
            # Проверяем доступность папки DICOM на запись (поддержка read-only)
            test_file_path = os.path.join(dicom_dir, f".write_test_{int(time.time())}")
            is_writable = False
            try:
                with open(test_file_path, "w") as f:
                    f.write("test")
                os.remove(test_file_path)
                is_writable = True
                output_dir = dicom_dir
            except Exception:
                is_writable = False

            if not is_writable:
                QMessageBox.warning(
                    self,
                    "Папка защищена от записи ⚠️",
                    "Выбранная папка защищена от записи (например, сетевой диск только для чтения).\n\n"
                    "Пожалуйста, выберите альтернативную папку для сохранения готовых RTSTRUCT файлов."
                )
                
                default_alt = getattr(self, "last_alternative_output_dir", "")
                if not default_alt or not os.path.isdir(default_alt):
                    default_alt = str(Path.home())
                    
                alt_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения", default_alt)
                if not alt_dir:
                    return
                
                output_dir = alt_dir
                self.last_alternative_output_dir = alt_dir
                self.save_settings()
                
            selected_organs = []
            for i in range(self.organs_list.count()):
                item = self.organs_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, dict):
                    org = data.get("name") or list(data.keys())[0] if data else ""
                else:
                    org = data
                if org == "header" or not org:
                    continue
                if item.checkState() == Qt.CheckState.Checked:
                    if org not in selected_organs:
                        selected_organs.append(org)
                    
            if not selected_organs:
                QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одного органа для сегментирования!")
                return
                
            if self.radio_merge_new.isChecked():
                merge_mode = "new"
            else:
                merge_mode = "merge"
                if hasattr(self, 'merge_rtstruct_combo') and self.merge_rtstruct_combo.count() > 0:
                    self.existing_rtstruct_path = self.merge_rtstruct_combo.currentData()

            # Блокируем интерфейс
            self.set_ui_enabled(False)
            self.log_edit.clear()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.scan_timer.stop()
            
            try:
                preset_name = self.preset_combo.currentText()
                preset_key = "abdominal_oar"
                if "Thorax" in preset_name or "Грудная" in preset_name:
                    preset_key = "thoracic_oar"
                elif "Pelvis" in preset_name or "Малый" in preset_name:
                    preset_key = "pelvis_oar"
                elif "Head & Neck" in preset_name or "Голова" in preset_name:
                    preset_key = "head_neck_oar"
                elif "Brachytherapy" in preset_name or "Брахитерапия" in preset_name:
                    preset_key = "brachytherapy_oar"
                else:
                    preset_key = "all"

                # Точность
                precision_modes = ["normal", "fast", "faster"]
                precision_mode = precision_modes[self.precision_combo.currentIndex()]

                # Сглаживание
                smoothing_sigmas = [0.5, 1.0, 1.5, 2.0]
                smoothing_sigma = smoothing_sigmas[self.smoothing_combo.currentIndex()] if self.smoothing_check.isChecked() else 0.0

                server_url = self.server_url_edit.text().strip()
                client_name = self.client_name_edit.text().strip()

                # Создаем и запускаем поток вычислений
                self.worker = SegmentationWorker(
                    server_url=server_url,
                    client_name=client_name,
                    dicom_dir=dicom_dir,
                    output_dir=output_dir,
                    preset_name=preset_key,
                    precision_mode=precision_mode,
                    selected_organs=selected_organs,
                    merge_mode=merge_mode,
                    existing_rtstruct_path=self.existing_rtstruct_path,
                    use_gpu=self.radio_gpu.isChecked(),
                    remove_blobs=self.clean_blobs_check.isChecked(),
                    smoothing_sigma=smoothing_sigma
                )
                self.worker.finished_signal.connect(self.on_segmentation_finished)
                self.worker.step_signal.connect(self.on_step_changed)
                self.worker.progress_signal.connect(self.progress_bar.setValue)
                self.worker.eta_signal.connect(self.on_eta_updated)
                self.worker.log_signal.connect(self.append_log)
                
                self.current_step_base_text = "Отправка на сервер..."
                self.spinner_index = 0
                self.pulse_tick = 0
                self.activity_timer.start()
                
                self.worker.start()
            except Exception as e:
                logger.exception("Ошибка при запуске сегментации")
                QMessageBox.critical(self, "Ошибка запуска", f"Не удалось инициализировать сегментацию:\n{e}")
                self.set_ui_enabled(True)
                self.scan_timer.start(15000)

        def set_ui_enabled(self, enabled: bool):
            self.input_edit.setEnabled(enabled)
            self.series_table.setEnabled(enabled)
            self.btn_input.setEnabled(enabled)
            # Кнопки выбора органов и пресетов должны быть заблокированы в режиме просмотра
            is_view_mode = hasattr(self, 'chk_show_structures') and self.chk_show_structures.isChecked()
            self.btn_select_all.setEnabled(enabled and not is_view_mode)
            self.btn_deselect_all.setEnabled(enabled and not is_view_mode)
            self.preset_combo.setEnabled(enabled and not is_view_mode)
            self.organs_list.setEnabled(enabled)
            self.precision_combo.setEnabled(enabled)
            self.clean_blobs_check.setEnabled(enabled)
            self.smoothing_check.setEnabled(enabled)
            if enabled:
                self.smoothing_combo.setEnabled(self.smoothing_check.isChecked())
                self.on_organ_selection_changed()
            else:
                self.smoothing_combo.setEnabled(False)
                
            self.color_preset_combo.setEnabled(enabled)
            
            # Блокировка переключателей устройства
            self.radio_cpu.setEnabled(enabled)
            self.radio_gpu.setEnabled(enabled)
            
            # Блокировка/восстановление переключателей слияния RTSTRUCT
            if enabled:
                row = self.series_table.currentRow()
                if row >= 0:
                    str_status = self.series_table.item(row, 2).text()
                    has_structs = (str_status != "Нет" and str_status != "No" and "0 " not in str_status)
                    self.radio_merge_new.setEnabled(True)
                    self.radio_merge_merge.setEnabled(has_structs)
                    self.update_merge_combo_state()
                else:
                    self.radio_merge_new.setEnabled(True)
                    self.radio_merge_merge.setEnabled(False)
                    self.update_merge_combo_state()
            else:
                self.radio_merge_new.setEnabled(False)
                self.radio_merge_merge.setEnabled(False)
                if hasattr(self, 'merge_rtstruct_combo'):
                    self.merge_rtstruct_combo.setEnabled(False)
            
            self.btn_run.setEnabled(True)
            if enabled:
                self.btn_run.setText("ОТПРАВИТЬ В ОЧЕРЕДЬ НА СЕРВЕР 🚀")
                self.btn_run.setStyleSheet("")
            else:
                self.btn_run.setText("ОТМЕНИТЬ РАСЧЕТ ❌")
                self.btn_run.setStyleSheet("""
                    QPushButton#btnRun {
                        background-color: #c0392b;
                        color: white;
                        font-weight: bold;
                        border: 1px solid #962d22;
                        font-size: 14px;
                        padding: 12px;
                        border-radius: 6px;
                    }
                    QPushButton#btnRun:hover {
                        background-color: #e74c3c;
                        border: 1px solid #c0392b;
                    }
                    QPushButton#btnRun:pressed {
                        background-color: #962d22;
                    }
                """)

        def update_activity_animation(self):
            self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
            spinner_char = self.SPINNER_FRAMES[self.spinner_index]
            self.status_step_label.setText(f"{self.current_step_base_text} {spinner_char}")
            
            self.pulse_tick += 1
            factor = (math.sin(self.pulse_tick * 0.15) + 1.0) / 2.0
            g = int(122 + (229 - 122) * factor)
            b = int(204 + (255 - 204) * factor)
            
            self.status_step_label.setStyleSheet(
                f"color: rgb(0, {g}, {b}); font-weight: bold; font-style: italic;"
            )

        def on_segmentation_finished(self, success: bool, message: str):
            try:
                # Фиксация статистики запуска
                elapsed = 0.0
                if self.worker and hasattr(self.worker, "_start_time"):
                    elapsed = time.time() - self.worker._start_time

                organs = self.worker.selected_organs if self.worker else []
                preset = self.preset_combo.currentText()
                precision = self.precision_combo.currentText()

                if success:
                    self.stats_manager.record_run(
                        status="success",
                        elapsed_seconds=elapsed,
                        organs_contoured=organs,
                        preset_name=preset,
                        precision_mode=precision
                    )
                else:
                    is_cancelled = False
                    if self.worker and getattr(self.worker, "is_cancelled", False):
                        is_cancelled = True
                    elif "отмен" in message.lower():
                        is_cancelled = True
                    
                    status_str = "cancelled" if is_cancelled else "failed"
                    self.stats_manager.record_run(
                        status=status_str,
                        elapsed_seconds=elapsed,
                        organs_contoured=[],
                        preset_name=preset,
                        precision_mode=precision
                    )
                
                # Обновляем интерфейс статистики
                if hasattr(self, "update_statistics_ui"):
                    self.update_statistics_ui()

                self.scan_timer.start(15000)
                self.set_ui_enabled(True)
                self.progress_bar.setRange(0, 100)
                self.activity_timer.stop()
                self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
                # Сбрасываем ETA-метку сразу после завершения
                self.eta_label.setText("")
                
                if self.sound_check.isChecked():
                    try:
                        import winsound
                        if success:
                            # Красивый восходящий мажорный аккорд (C5 -> E5 -> G5)
                            winsound.Beep(523, 150)
                            time.sleep(0.05)
                            winsound.Beep(659, 150)
                            time.sleep(0.05)
                            winsound.Beep(784, 250)
                        else:
                            # Низкий предупреждающий звук (A3)
                            winsound.Beep(220, 500)
                    except Exception as e:
                        logger.error(f"Не удалось воспроизвести звуковое оповещение: {e}")
                
                if success:
                    self.progress_bar.setValue(100)
                    self.status_step_label.setText("Текущий шаг: Готово!")
                    
                    # Парсинг количества структур (из текста сообщения)
                    count = 0
                    time_str = "0.0"
                    match_count = re.search(r'добавлено структур:\s*(\d+)', message.lower())
                    if match_count:
                        count = match_count.group(1)
                    
                    match_time = re.search(r'время работы:\s*([\d\.]+)', message.lower())
                    if match_time:
                        time_str = match_time.group(1)

                    final_log = f"[INFO]: Пайплайн успешно завершен! Добавлено структур: {count}. Общее время работы: {time_str} сек."
                    self.log_edit.append(f"<br><span style='background-color: #107c41; color: white; font-weight: bold; padding: 4px;'>{final_log}</span><br>")
                    
                    # Немедленно обновляем интерфейс, чтобы подтянуть созданный RTSTRUCT
                    selected = self.series_table.selectedItems()
                    if selected:
                        row = selected[0].row()
                        path_item = self.series_table.item(row, 6)
                        if path_item:
                            selected_path = path_item.text()
                            # Сканируем RTSTRUCT
                            self.check_for_rtstruct(selected_path)
                            
                            # Обновляем статус структуры в таблице на точное количество найденных файлов
                            item_str = self.series_table.item(row, 2)
                            if item_str:
                                item_str.setText(format_rtstruct_count(len(self.rtstruct_files)))
                            
                            # Автоматически активируем галочку и отрисовываем контуры во вьюере
                            if hasattr(self, 'chk_show_structures') and self.chk_show_structures.isEnabled():
                                self.chk_show_structures.setChecked(True)
                    
                    QTimer.singleShot(100, lambda: QMessageBox.information(self, "Успех", "Автоматическое оконтурирование завершено успешно!"))
                    # Сбрасываем прогресс-бар до 0 через 5 секунд, чтобы не висел при просмотре снимков
                    QTimer.singleShot(5000, lambda: self.progress_bar.setValue(0))
                else:
                    self.progress_bar.setValue(0)
                    if "отменена пользователем" in message.lower() or "отменен пользователем" in message.lower():
                        self.status_step_label.setText("Текущий шаг: Расчет отменен!")
                        self.status_step_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                        QTimer.singleShot(100, lambda: QMessageBox.warning(self, "Предупреждение", "Процесс оконтурирования был прерван."))
                    else:
                        self.status_step_label.setText("Текущий шаг: Ошибка!")
                        QTimer.singleShot(100, lambda msg=message: QMessageBox.critical(self, "Критическая ошибка", f"Произошел сбой при сегментации:\n{msg}"))
                    # Сбрасываем прогресс-бар через 3 секунды в обоих случаях (отмена/ошибка)
                    QTimer.singleShot(3000, lambda: self.progress_bar.setValue(0))
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Критическая ошибка в on_segmentation_finished: {e}")
                QTimer.singleShot(100, lambda err=e: QMessageBox.critical(self, "Сбой GUI", f"Ошибка в on_segmentation_finished:\n{err}"))

        def on_step_changed(self, step_text: str):
            self.current_step_base_text = step_text
            self.status_step_label.setText(f"{step_text} {self.SPINNER_FRAMES[self.spinner_index]}")

        def on_eta_updated(self, elapsed: float, eta: float):
            """Обновляет метку ETA во время расчёта ИИ."""
            def fmt(s: float) -> str:
                m = int(s // 60)
                sec = int(s % 60)
                return f"{m} мин {sec:02d} сек" if m > 0 else f"{sec} сек"
            txt = f"⏱ Прошло: {fmt(elapsed)}"
            if eta > 0:
                txt += f"  |  Ожидается ещё: ~{fmt(eta)}"
            self.eta_label.setText(txt)

        def on_tab_changed(self, index):
            if self.tab_widget.tabText(index) == "📊 Статистика":
                self.update_statistics_ui()

        def create_statistics_tab(self):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(6, 4, 6, 4)
            layout.setSpacing(4)
            
            header = QLabel("📊 Статистика автооконтурирований")
            header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            layout.addWidget(header)
            
            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(6)
            
            def create_stat_card(title: str, val: str, color: str = "#ffffff"):
                card = QFrame()
                card.setStyleSheet("""
                    QFrame {
                        background-color: #242424;
                        border: 1px solid #333333;
                        border-radius: 6px;
                        padding: 4px;
                    }
                """)
                card_lay = QVBoxLayout(card)
                card_lay.setContentsMargins(4, 4, 4, 4)
                card_lay.setSpacing(2)
                
                title_lbl = QLabel(title)
                title_lbl.setStyleSheet("font-size: 9px; color: #888888; font-weight: bold;")
                val_lbl = QLabel(val)
                val_lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")
                
                card_lay.addWidget(title_lbl)
                card_lay.addWidget(val_lbl)
                return card, val_lbl
                
            self.card_total, self.lbl_stat_total = create_stat_card("ВСЕГО ЗАПУСКОВ", "0")
            self.card_success, self.lbl_stat_success = create_stat_card("УСПЕШНО", "0", "#2ecc71")
            self.card_fail, self.lbl_stat_fail = create_stat_card("СБОЕВ / ОТМЕН", "0 / 0", "#e74c3c")
            self.card_organs, self.lbl_stat_organs = create_stat_card("ОКОНТУРЕНО OAR", "0", "#3498db")
            
            grid_layout.addWidget(self.card_total, 0, 0)
            grid_layout.addWidget(self.card_success, 0, 1)
            grid_layout.addWidget(self.card_fail, 1, 0)
            grid_layout.addWidget(self.card_organs, 1, 1)
            layout.addWidget(grid_widget)
            
            time_widget = QFrame()
            time_widget.setStyleSheet("""
                QFrame {
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            time_lay = QHBoxLayout(time_widget)
            time_lay.setContentsMargins(10, 5, 10, 5)
            
            self.lbl_stat_total_time = QLabel("Общее время работы: 0 сек")
            self.lbl_stat_total_time.setStyleSheet("font-size: 12px; color: #a0a0a0;")
            self.lbl_stat_avg_time = QLabel("Среднее время запуска: 0 сек")
            self.lbl_stat_avg_time.setStyleSheet("font-size: 12px; color: #a0a0a0;")
            
            time_lay.addWidget(self.lbl_stat_total_time)
            time_lay.addStretch()
            time_lay.addWidget(self.lbl_stat_avg_time)
            layout.addWidget(time_widget)
            
            table_lbl = QLabel("🕒 Последние запуски:")
            table_lbl.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 4px; margin-bottom: 2px;")
            layout.addWidget(table_lbl)
            
            self.stats_table = QTableWidget()
            self.stats_table.setColumnCount(5)
            self.stats_table.setHorizontalHeaderLabels(["Время", "Пресет", "Режим", "Длительность", "Статус"])
            self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.stats_table.setStyleSheet("""
                QTableWidget {
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    gridline-color: #2d2d2d;
                    border-radius: 6px;
                }
                QHeaderView::section {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    padding: 4px;
                    border: 0px;
                    font-weight: bold;
                }
            """)
            self.stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.stats_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.stats_table.setMaximumHeight(180)
            layout.addWidget(self.stats_table)
            
            organs_lbl = QLabel("🔝 Популярные органы:")
            organs_lbl.setStyleSheet("font-weight: bold; color: #ffffff; margin-top: 4px; margin-bottom: 2px;")
            layout.addWidget(organs_lbl)
            
            self.stats_organs_list = QListWidget()
            self.stats_organs_list.setStyleSheet("""
                QListWidget {
                    background-color: #1e1e1e;
                    border: 1px solid #2d2d2d;
                    border-radius: 6px;
                }
            """)
            self.stats_organs_list.setMaximumHeight(140)
            layout.addWidget(self.stats_organs_list)
            
            btn_lay = QHBoxLayout()
            self.btn_stats_refresh = QPushButton("🔄 Обновить")
            self.btn_stats_refresh.clicked.connect(self.update_statistics_ui)
            
            self.btn_stats_reset = QPushButton("🗑️ Сбросить статистику")
            self.btn_stats_reset.setStyleSheet("""
                QPushButton {
                    background-color: #2c1a1a;
                    border: 1px solid #5a2a2a;
                    color: #ff6b6b;
                }
                QPushButton:hover {
                    background-color: #4a2a2a;
                }
            """)
            self.btn_stats_reset.clicked.connect(self.reset_statistics_with_password)
            
            btn_lay.addWidget(self.btn_stats_refresh)
            btn_lay.addWidget(self.btn_stats_reset)
            layout.addLayout(btn_lay)
            
            scroll.setWidget(widget)
            return scroll

        def update_statistics_ui(self):
            if not hasattr(self, 'lbl_stat_total'):
                return
            try:
                stats = self.stats_manager.get_stats()
                
                total = stats.get("total_runs", 0)
                success = stats.get("successful_runs", 0)
                fail = stats.get("failed_runs", 0)
                cancelled = stats.get("cancelled_runs", 0)
                organs_cnt = stats.get("total_organs_contoured", 0)
                elapsed = stats.get("total_elapsed_time_seconds", 0.0)
                
                self.lbl_stat_total.setText(str(total))
                self.lbl_stat_success.setText(str(success))
                self.lbl_stat_fail.setText(f"{fail} / {cancelled}")
                self.lbl_stat_organs.setText(str(organs_cnt))
                
                if elapsed >= 3600:
                    h = int(elapsed // 3600)
                    m = int((elapsed % 3600) // 60)
                    s = int(elapsed % 60)
                    time_str = f"Общее время работы: {h} ч {m} м {s} с"
                elif elapsed >= 60:
                    m = int(elapsed // 60)
                    s = int(elapsed % 60)
                    time_str = f"Общее время работы: {m} м {s} с"
                else:
                    time_str = f"Общее время работы: {round(elapsed, 1)} сек"
                    
                self.lbl_stat_total_time.setText(time_str)
                
                avg = elapsed / success if success > 0 else 0.0
                self.lbl_stat_avg_time.setText(f"Среднее время запуска: {round(avg, 1)} сек")
                
                self.stats_table.setRowCount(0)
                recent = stats.get("recent_runs", [])
                for run in recent:
                    row = self.stats_table.rowCount()
                    self.stats_table.insertRow(row)
                    
                    status_lbl = "Успех"
                    color_str = "#2ecc71"
                    if run.get("status") == "cancelled":
                        status_lbl = "Отмена"
                        color_str = "#f1c40f"
                    elif run.get("status") == "failed":
                        status_lbl = "Ошибка"
                        color_str = "#e74c3c"
                        
                    status_item = QTableWidgetItem(status_lbl)
                    status_item.setForeground(QColor(color_str))
                    status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    time_item = QTableWidgetItem(str(run.get("timestamp", "")))
                    time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    preset_item = QTableWidgetItem(str(run.get("preset", "")))
                    
                    precision_item = QTableWidgetItem(str(run.get("precision", "")))
                    precision_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    dur = run.get("elapsed_seconds", 0.0)
                    dur_item = QTableWidgetItem(f"{dur} сек")
                    dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    self.stats_table.setItem(row, 0, time_item)
                    self.stats_table.setItem(row, 1, preset_item)
                    self.stats_table.setItem(row, 2, precision_item)
                    self.stats_table.setItem(row, 3, dur_item)
                    self.stats_table.setItem(row, 4, status_item)
                
                self.stats_organs_list.clear()
                org_stats = stats.get("organ_stats", {})
                sorted_orgs = sorted(org_stats.items(), key=lambda x: x[1], reverse=True)
                
                for org, count in sorted_orgs:
                    ru_name = org
                    if hasattr(self.engine, "ru_names") and org in self.engine.ru_names:
                        ru_name = self.engine.ru_names[org]
                    self.stats_organs_list.addItem(f"• {ru_name}: {count} раз(а)")
                
                if not sorted_orgs:
                    self.stats_organs_list.addItem("Пока нет данных о сегментированных органах.")
                    
            except Exception as e:
                logger.error(f"Ошибка обновления статистики: {e}")

        def reset_statistics_with_password(self):
            from PyQt6.QtWidgets import QInputDialog, QLineEdit
            text, ok = QInputDialog.getText(
                self, 
                "Сброс статистики", 
                "Для подтверждения сброса введите пароль:", 
                QLineEdit.EchoMode.Password
            )
            if ok:
                if text == "rtp":
                    self.stats_manager.reset_stats()
                    self.update_statistics_ui()
                    QMessageBox.information(self, "Успех", "Статистика успешно сброшена.")
                else:
                    QMessageBox.critical(self, "Ошибка", "Неверный пароль. Сброс отклонен.")

        def show_help(self):
            dialog = QDialog(self)
            dialog.setWindowTitle("Справка и медицинский дисклеймер")
            dialog.setMinimumSize(640, 560)
            dialog.setStyleSheet(self.styleSheet())
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.setSpacing(12)
            
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
            <li><b>Динамические пресеты:</b> Вы можете легко добавлять или редактировать анатомические пресеты во внешних файлах в папке <span class="highlight">config/presets/</span> проекта.</li>
            <li><b>Интеллектуальное GPU-ускорение:</b> При наличии видеокарты Nvidia с поддержкой CUDA расчеты будут ускорены в 20-30 раз.</li>
            <li><b>3D Постобработка:</b> Очистка мелкого шума нейросети (Remove small blobs) и сглаживание Гаусса для сглаживания «ступенчатости» контуров.</li>
            <li><b>Кастомизация цветов:</b> Интерактивный выбор цветов структур кликом по органу в списке. Поддержка готовых цветовых палитр (QUANTEC, Неон).</li>
            <li><b>Пресет «Брахитерапия»:</b> Специальный пресет, содержащий мочевой пузырь, тонкий кишечник и сдублированную геометрическую маску кишки под двумя именами.</li>
        </ul>
    </div>

    <div class="disclaimer-box">
        <div class="disclaimer-title">⚠️ ВАЖНЫЙ МЕДИЦИНСКИЙ ДИСКЛЕЙМЕР</div>
        <p style="margin: 0; font-size: 12.5px; color: #e0b0b0;">
            Данное программное обеспечение предоставляется исключительно для научных и исследовательских целей (<b>Research Use Only</b>). <br><br>
            Автоматическая разметка <b>не является окончательной клинической разметкой</b>. Любая импортированная разметка <b>подлежит обязательному ручному контролю, валидации и коррекции</b> сертифицированным медицинским физиком или радиационным онкологом в системе планирования (TPS) перед облучением пациента.
        </p>
    </div>
</body>
</html>"""
            
            browser.setHtml(html_content)
            layout.addWidget(browser, 1)
            
            btn_close = QPushButton("Ясно, закрыть")
            btn_close.setObjectName("btnAction")
            btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_close.clicked.connect(dialog.accept)
            
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_layout.addWidget(btn_close)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)
            
            dialog.exec()

        def closeEvent(self, event):
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, 'Подтверждение выхода',
                "Вы действительно хотите выйти из программы?\nЕсли идет оконтуривание, оно может быть прервано.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                logging.info("Закрытие панели управления сервером...")
                if hasattr(self, "server_process") and self.server_process:
                    logging.info("Останавливаем фоновый процесс бэкенда сервера...")
                    pid = self.server_process.pid
                    if os.name == 'nt':
                        # На Windows используем taskkill /T для убийства всего дерева процессов (uvicorn + дочерние)
                        import subprocess as sp
                        sp.run(
                            f"taskkill /F /T /PID {pid}",
                            shell=True,
                            stdout=sp.DEVNULL,
                            stderr=sp.DEVNULL
                        )
                    else:
                        self.server_process.terminate()
                    try:
                        self.server_process.wait(timeout=3)
                    except Exception:
                        self.server_process.kill()
                if hasattr(self, "server_stdout_file") and self.server_stdout_file:
                    try:
                        self.server_stdout_file.close()
                    except Exception:
                        pass
                event.accept()
            else:
                event.ignore()


        def get_local_ip(self) -> str:
            """Получает локальный IP адрес сервера в сети."""
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                return "127.0.0.1"

        def get_server_url(self) -> str:
            return self.server_url_edit.text().strip().rstrip('/')

        def toggle_pause(self):
            """Переключает статус паузы очереди на сервере в фоновом потоке."""
            if getattr(self, "is_toggling_pause", False):
                return
                
            self.is_toggling_pause = True
            self.btn_pause_toggle.setEnabled(False)
            
            current_paused = self.server_is_paused
            next_status = "Включение..." if current_paused else "Приостановка..."
            self.btn_pause_toggle.setText(f"{next_status} 🔄")
            self.btn_pause_toggle.setStyleSheet("""
                QPushButton#btnPauseActive {
                    background-color: #7f8c8d;
                    border: 1px solid #95a5a6;
                    color: #ffffff;
                    padding: 8px 18px;
                    font-size: 13px;
                    font-weight: bold;
                }
            """)
            
            def worker():
                import requests
                server_url = self.get_server_url()
                url = f"{server_url}/api/server/resume" if current_paused else f"{server_url}/api/server/pause"
                success = False
                try:
                    res = requests.post(url, timeout=2.5)
                    if res.status_code == 200:
                        success = True
                except Exception as e:
                    logger.error(f"Не удалось отправить запрос смены паузы: {e}")
                    
                # Безопасно отправляем сигнал в главный поток
                self.pause_signaler.pause_signal.emit(success, current_paused)
                
            import threading
            threading.Thread(target=worker, daemon=True).start()

        def update_gui_after_pause(self, success, current_paused):
            """Вызывается строго в главном потоке GUI при получении сигнала от фонового worker-а."""
            self.is_toggling_pause = False
            self.btn_pause_toggle.setEnabled(True)
            if success:
                self.server_is_paused = not current_paused
                if self.server_is_paused:
                    self.btn_pause_toggle.setText("СЕРВЕР НА ПАУЗЕ ⏸️")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d35400, stop: 1 #a04000);
                            border: 1px solid #e67e22;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
                else:
                    self.btn_pause_toggle.setText("СЕРВЕР АКТИВЕН 🟢")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #27ae60, stop: 1 #1e8449);
                            border: 1px solid #2ecc71;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
            else:
                self.server_is_paused = current_paused
                # Бесшумный откат состояния кнопки при ошибке
                if self.server_is_paused:
                    self.btn_pause_toggle.setText("СЕРВЕР НА ПАУЗЕ ⏸️")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d35400, stop: 1 #a04000);
                            border: 1px solid #e67e22;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
                else:
                    self.btn_pause_toggle.setText("СЕРВЕР АКТИВЕН 🟢")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #27ae60, stop: 1 #1e8449);
                            border: 1px solid #2ecc71;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
                self.update_server_ui()

        def update_server_ui(self):
            """Вызывается по таймеру раз в секунду: запрашивает статус через REST API в фоновом потоке и обновляет интерфейс."""
            if getattr(self, "is_toggling_pause", False):
                return
            if getattr(self, "is_updating_server_ui", False):
                return
                
            self.is_updating_server_ui = True
            
            def worker():
                import requests
                success = False
                data = {}
                active_job_logs = []
                try:
                    server_url = self.get_server_url()
                    response = requests.get(f"{server_url}/api/server/status", timeout=1.5)
                    if response.status_code == 200:
                        data = response.json()
                        success = True
                        
                        info_list = data.get("jobs", [])
                        processing_job = None
                        for item in info_list:
                            if item.get("status") == "PROCESSING":
                                processing_job = item
                                break
                        
                        if processing_job:
                            active_job_id = processing_job.get("job_id")
                            job_status_res = requests.get(f"{server_url}/api/jobs/{active_job_id}/status", timeout=1.0)
                            if job_status_res.status_code == 200:
                                active_job_logs = job_status_res.json().get("logs", [])
                except Exception as e:
                    logger.debug(f"Ошибка фонового обновления статуса сервера: {e}")
                
                # Безопасно передаем результаты в главный поток GUI через сигнал
                self.server_status_signaler.status_signal.emit(success, data, active_job_logs)
                
            import threading
            threading.Thread(target=worker, daemon=True).start()

        def handle_server_status_update(self, success: bool, data: dict, active_job_logs: list):
            """Вызывается строго в главном потоке GUI для безопасного обновления UI."""
            self.is_updating_server_ui = False
            if getattr(self, "is_toggling_pause", False):
                return
                
            if success:
                is_paused = data.get("is_paused", False)
                info_list = data.get("jobs", [])
                
                # Обновляем локальное состояние паузы
                self.server_is_paused = is_paused
                if is_paused:
                    self.btn_pause_toggle.setText("СЕРВЕР НА ПАУЗЕ ⏸️")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d35400, stop: 1 #a04000);
                            border: 1px solid #e67e22;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
                else:
                    self.btn_pause_toggle.setText("СЕРВЕР АКТИВЕН 🟢")
                    self.btn_pause_toggle.setStyleSheet("""
                        QPushButton#btnPauseActive {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #27ae60, stop: 1 #1e8449);
                            border: 1px solid #2ecc71;
                            color: #ffffff;
                            padding: 8px 18px;
                            font-size: 13px;
                            font-weight: bold;
                        }
                    """)
                    
                self.lbl_server_address.setText(f"API запущен: http://{self.get_local_ip()}:8000")
                
                # Обновление таблицы очереди
                self.table_queue.setRowCount(len(info_list))
                from PyQt6.QtGui import QBrush, QColor, QFont
                for row, item in enumerate(info_list):
                    self.table_queue.setItem(row, 0, self.create_table_item(item["client_name"]))
                    self.table_queue.setItem(row, 1, self.create_table_item(item["patient_name"]))
                    self.table_queue.setItem(row, 2, self.create_table_item(item["patient_id"]))
                    self.table_queue.setItem(row, 3, self.create_table_item(item["preset"]))
                    
                    status_item = self.create_table_item(item["status"], centered=True)
                    if item["status"] == "SUCCESS":
                        status_item.setForeground(QBrush(QColor("#2ecc71")))
                    elif item["status"] == "FAILED":
                        status_item.setForeground(QBrush(QColor("#e74c3c")))
                    elif item["status"] == "CANCELLED":
                        status_item.setForeground(QBrush(QColor("#f39c12")))
                    elif item["status"] == "PROCESSING":
                        status_item.setForeground(QBrush(QColor("#3498db")))
                        status_item.setFont(QFont("Segoe UI", weight=QFont.Weight.Bold))
                    self.table_queue.setItem(row, 4, status_item)
                    
                    prog_text = f"{item['progress']}%" if item["status"] in ["PROCESSING", "SUCCESS"] else "-"
                    self.table_queue.setItem(row, 5, self.create_table_item(prog_text, centered=True))
                    
                    self.table_queue.setItem(row, 6, self.create_table_item(item["created_at"], centered=True))
                    self.table_queue.item(row, 0).setData(Qt.ItemDataRole.UserRole, item["job_id"])
                    
                # Прогресс-бар и ETA
                local_running = hasattr(self, 'worker') and self.worker and self.worker.isRunning()
                if not local_running:
                    processing_job = None
                    for item in info_list:
                        if item.get("status") == "PROCESSING":
                            processing_job = item
                            break
                            
                    if processing_job:
                        prog_val = int(processing_job.get("progress", 0))
                        self.progress_bar.setValue(prog_val)
                        self.progress_bar.setRange(0, 100)
                        
                        step_text = processing_job.get("current_step", "Выполнение...")
                        self.status_step_label.setText(f"Текущий шаг (Сеть): {step_text}")
                        self.status_step_label.setStyleSheet("color: #3498db; font-weight: bold; font-style: italic;")
                        
                        elapsed = processing_job.get("elapsed", 0.0)
                        eta = processing_job.get("eta", 0.0)
                        def fmt(s: float) -> str:
                            m = int(s // 60)
                            sec = int(s % 60)
                            return f"{m} мин {sec:02d} сек" if m > 0 else f"{sec} сек"
                        txt = f"⏱ Прошло (Сеть): {fmt(elapsed)}"
                        if eta > 0:
                            txt += f"  |  Ожидается ещё: ~{fmt(eta)}"
                        self.eta_label.setText(txt)
                        
                        active_job_id = processing_job.get("job_id")
                        if not hasattr(self, '_current_active_job_id') or self._current_active_job_id != active_job_id:
                            self._current_active_job_id = active_job_id
                            self.last_server_log_index = 0
                            patient_name = processing_job.get("patient_name", "Неизвестный")
                            self.append_log(f"<br><span style='background-color: #34495e; color: white; font-weight: bold; padding: 6px;'>=== Логи автооконтурирования: {patient_name} ===</span><br>", "#3498db")
                        
                        if len(active_job_logs) > self.last_server_log_index:
                            for new_line in active_job_logs[self.last_server_log_index:]:
                                color = "#a0a0a2"
                                if "ERROR" in new_line or "Exception" in new_line or "failed" in new_line.lower():
                                    color = "#ff6b6b"
                                elif "WARNING" in new_line:
                                    color = "#f1c40f"
                                elif "шаг" in new_line.lower() or "---" in new_line:
                                    color = "#3498db"
                                elif "totalsegmentator" in new_line.lower():
                                    color = "#2ecc71"
                                self.append_log(new_line, color)
                            self.last_server_log_index = len(active_job_logs)
                    else:
                        if "(Сеть)" in self.status_step_label.text():
                            self.progress_bar.setValue(0)
                            self.status_step_label.setText("Текущий шаг: Ожидание запуска...")
                            self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
                            self.eta_label.setText("")
                            
                            if hasattr(self, '_current_active_job_id') and self._current_active_job_id:
                                final_log = "[INFO]: Сетевой пайплайн успешно завершен!"
                                self.log_edit.append(f"<br><span style='background-color: #107c41; color: white; font-weight: bold; padding: 4px;'>{final_log}</span><br>")
                                self._current_active_job_id = None
                                self.last_server_log_index = 0
                            self.update_statistics_ui()
            else:
                self.lbl_server_address.setText("Подключение к API серверу... (Запуск/Оффлайн)")
                self.table_queue.setRowCount(0)

        def create_table_item(self, text: str, centered: bool = False) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            if centered:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return item

        def show_context_menu(self, pos):
            """Отображает контекстное меню для управления задачами в очереди."""
            row = self.table_queue.currentRow()
            if row < 0:
                return
                
            # Извлекаем job_id задачи
            job_id = self.table_queue.item(row, 0).data(Qt.ItemDataRole.UserRole)
            status_text = self.table_queue.item(row, 4).text()
                
            menu = QMenu(self)
            
            # Настройка действий
            cancel_action = menu.addAction("Отменить задачу ❌")
            cancel_action.setEnabled(status_text in ["PENDING", "PROCESSING"])
            
            # Приоритет
            prioritize_action = menu.addAction("Поднять в начало очереди ⬆️")
            is_pending = (status_text == "PENDING")
            prioritize_action.setEnabled(is_pending and row > 0)
            
            # Возобновление
            resume_action = menu.addAction("Возобновить задачу 🔄")
            resume_action.setEnabled(status_text in ["FAILED", "CANCELLED"])
            
            # Действия
            action = menu.exec(self.table_queue.mapToGlobal(pos))
            if action == cancel_action:
                self.cancel_job_by_id(job_id)
            elif action == prioritize_action:
                self.prioritize_job(job_id)
            elif action == resume_action:
                self.resume_job_by_id(job_id)

        def cancel_job_by_id(self, job_id: str):
            reply = QMessageBox.question(
                self, "Подтверждение отмены",
                f"Вы действительно хотите отменить выбранную задачу?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    import requests
                    server_url = self.get_server_url()
                    res = requests.delete(f"{server_url}/api/jobs/{job_id}/cancel", timeout=3)
                    if res.status_code != 200:
                        raise RuntimeError(res.text)
                except Exception as e:
                    logger.error(f"Не удалось отменить задачу {job_id}: {e}")
                    QMessageBox.warning(self, "Ошибка отмены", f"Не удалось отменить выбранную задачу: {e}")

        def prioritize_job(self, job_id: str):
            """Поднимает задачу на самый верх очереди (показывает приоритет)."""
            try:
                import requests
                server_url = self.get_server_url()
                res = requests.post(f"{server_url}/api/jobs/{job_id}/prioritize", timeout=3)
                if res.status_code != 200:
                    raise RuntimeError(res.text)
                logger.info(f"Задача {job_id} успешно приоритезирована.")
            except Exception as e:
                logger.error(f"Не удалось изменить приоритет задачи {job_id}: {e}")
                QMessageBox.warning(self, "Ошибка изменения приоритета", f"Не удалось изменить приоритет: {e}")

        def send_new_queue_order(self, job_ids: list):
            """Отправляет обновленный порядок задач в очереди на бэкенд."""
            try:
                import requests
                server_url = self.get_server_url()
                res = requests.post(
                    f"{server_url}/api/queue/reorder",
                    json={"job_ids": job_ids},
                    timeout=3
                )
                if res.status_code != 200:
                    raise RuntimeError(res.text)
                logger.info("Новый порядок очереди успешно отправлен на сервер.")
            except Exception as e:
                logger.error(f"Не удалось отправить новый порядок очереди на сервер: {e}")
                QMessageBox.warning(self, "Ошибка сортировки", f"Не удалось изменить порядок задач: {e}")

        def resume_job_by_id(self, job_id: str):
            """Запрос на возобновление упавшей или отмененной задачи."""
            try:
                import requests
                server_url = self.get_server_url()
                res = requests.post(f"{server_url}/api/jobs/{job_id}/resume", timeout=3)
                if res.status_code != 200:
                    try:
                        detail = res.json().get("detail", res.text)
                    except Exception:
                        detail = res.text
                    raise RuntimeError(detail)
                logger.info(f"Задача {job_id} успешно возобновлена.")
            except Exception as e:
                logger.error(f"Не удалось возобновить задачу {job_id}: {e}")
                QMessageBox.warning(self, "Ошибка возобновления", f"Не удалось возобновить выбранную задачу: {e}")


# ==============================================================================
# Исполняемый блок программы
# ==============================================================================

if __name__ == "__main__":
    if not PYQT_AVAILABLE:
        print("Ошибка: Для запуска GUI необходима библиотека PyQt6.")
        print("Установите ее: pip install PyQt6")
        sys.exit(1)

    # Регистрируем уникальный AppUserModelID — это ключ для отображения
    # собственной иконки на панели задач Windows (не иконки python.exe).
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AIContour.AutoContour.1.0")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Настройка палитры приложения для серых неактивных пунктов меню на Windows
    palette = app.palette()
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#666666"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#666666"))
    app.setPalette(palette)

    # Защита от запуска второй копии программы (используем мьютекс Windows)
    try:
        import ctypes
        mutex_name = "AIContourServerMutex_1.0"
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
        last_error = ctypes.windll.kernel32.GetLastError()
        if last_error == 183:  # ERROR_ALREADY_EXISTS
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "Ошибка запуска", "Программа уже запущена!")
            sys.exit(0)
    except Exception:
        pass

    # Иконка на QApplication охватывает все окна приложения
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
