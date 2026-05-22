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
2. Динамическое управление пресетами и цветами из presets.json.
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
        QProgressDialog
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject, QSettings, QTimer
    from PyQt6.QtGui import QTextCursor, QBrush, QColor, QFont, QIcon, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Импортируем вычислительный движок
try:
    from contour_engine import ContourEngine
except ImportError:
    # На случай запуска без движка
    ContourEngine = None

# Импортируем конфигурационные данные
try:
    from config import ORGAN_GROUPS, EXTERNAL_ALIASES
except ImportError:
    ORGAN_GROUPS = {}
    EXTERNAL_ALIASES = {}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AutoContourGUI")


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
        """Поток для вычислений сегментации TotalSegmentator, чтобы GUI не зависал."""
        finished_signal = pyqtSignal(bool, str)
        step_signal = pyqtSignal(str)
        progress_signal = pyqtSignal(int)
        eta_signal = pyqtSignal(float, float)  # (elapsed_sec, eta_sec)

        def __init__(
            self,
            engine: ContourEngine,
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
            self.engine = engine
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
            self.process = None

        def cancel(self):
            self.is_cancelled = True
            if self.process and self.process.poll() is None:
                try:
                    logger.info("Отмена: принудительное завершение процесса TotalSegmentator...")
                    self.process.kill()
                except Exception as e:
                    logger.error(f"Не удалось принудительно завершить процесс: {e}")

        def run(self):
            try:
                self._start_time = time.time()

                def callback(step_text: str):
                    self.step_signal.emit(step_text)
                    
                def prog_callback(val: int, text: str):
                    self.progress_signal.emit(val)
                    self.step_signal.emit(text)
                    # Расчёт ETA: если уже есть прогресс > 2%, прогнозируем оставшееся время
                    if val > 2:
                        elapsed = time.time() - self._start_time
                        eta = (elapsed / val) * (100 - val)
                        self.eta_signal.emit(elapsed, eta)
                    
                def reg_proc(p):
                    self.process = p
                    
                def is_canc():
                    return self.is_cancelled

                added, elapsed = self.engine.run_pipeline(
                    dicom_dir_path=self.dicom_dir,
                    output_dir_path=self.output_dir,
                    preset_name=self.preset_name,
                    precision_mode=self.precision_mode,
                    selected_organs=self.selected_organs,
                    merge_mode=self.merge_mode,
                    existing_rtstruct_path=self.existing_rtstruct_path,
                    use_gpu=self.use_gpu,
                    remove_blobs=self.remove_blobs,
                    smoothing_sigma=self.smoothing_sigma,
                    step_callback=callback,
                    progress_callback=prog_callback,
                    is_cancelled_cb=is_canc,
                    register_process_cb=reg_proc
                )
                if self.is_cancelled:
                    self.finished_signal.emit(False, "Операция отменена пользователем.")
                else:
                    msg = f"Пайплайн успешно завершен! Добавлено структур: {added}. Общее время работы: {elapsed:.1f} сек."
                    self.finished_signal.emit(True, msg)
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
        font-weight: bold;
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

    class MainWindow(QMainWindow):
        """Главное окно графического интерфейса приложения."""
        def __init__(self):
            super().__init__()
            self.setWindowTitle("AI Contour - Автооконтурирование КТ органов риска")
            self.setMinimumSize(960, 760)
            self.showMaximized()
            self.existing_rtstruct_path = None
            self.is_updating_presets = False
            self.worker = None
            self.settings = QSettings("AIContourCorp", "AIContour")

            # Инициализация вычислительного движка
            self.engine = ContourEngine()

            # Настройка перенаправления логов в реальном времени
            self.log_signaler = LogSignaler()
            self.log_signaler.log_signal.connect(self.append_log)
            self.log_handler = QTextEditLogHandler(self.log_signaler)
            logging.getLogger().addHandler(self.log_handler)

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

            self.init_ui()
            self.load_settings()

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
            splitter = QSplitter(Qt.Orientation.Horizontal)
            main_layout.addWidget(splitter, 1)

            # --- ЛЕВАЯ КОЛОНКА (Вкладки настроек) ---
            left_card = QFrame()
            left_card.setObjectName("card")
            left_card.setMinimumWidth(400)
            left_card.setMaximumWidth(480)
            left_layout = QVBoxLayout(left_card)
            left_layout.setContentsMargins(5, 5, 5, 5)

            self.tab_widget = QTabWidget()
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
            self.preset_combo = QComboBox()
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

            tab1_layout.addWidget(self.organs_header)
            tab1_layout.addWidget(self.organs_list)
            
            # Двойной клик по элементу списка для выбора цвета
            self.organs_list.itemDoubleClicked.connect(self.pick_organ_color)
            
            self.tab_widget.addTab(tab1_widget, "🎯 Контуры и снимки")

            # ------------------------------------------------------------------
            # ВКЛАДКА 2: Параметры ИИ и Цвета
            # ------------------------------------------------------------------
            tab2_widget = QWidget()
            tab2_layout = QVBoxLayout(tab2_widget)
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
            
            # Группа: Работа с существующими контурами (перенесено из Tab 1)
            merge_group = QGroupBox("Работа с существующими контурами")
            merge_group_layout = QVBoxLayout(merge_group)
            merge_group_layout.setSpacing(10)
            
            self.status_rtstruct_label = QLabel("RTSTRUCT: путь не выбран")
            self.status_rtstruct_label.setStyleSheet("color: #888888;")
            self.status_rtstruct_label.setWordWrap(True)
            merge_group_layout.addWidget(self.status_rtstruct_label)
            
            self.merge_btn_group = QButtonGroup(self)
            self.radio_merge_new = QRadioButton("Создать новый файл RTSTRUCT")
            self.radio_merge_merge = QRadioButton("Дополнить существующий файл")
            self.radio_merge_overwrite = QRadioButton("Перезаписать существующий файл")
            
            self.radio_merge_merge.setChecked(True)
            self.merge_btn_group.addButton(self.radio_merge_new, 1)
            self.merge_btn_group.addButton(self.radio_merge_merge, 2)
            self.merge_btn_group.addButton(self.radio_merge_overwrite, 3)
            
            merge_group_layout.addWidget(self.radio_merge_new)
            merge_group_layout.addWidget(self.radio_merge_merge)
            merge_group_layout.addWidget(self.radio_merge_overwrite)
            tab2_layout.addWidget(merge_group)
            
            gpu_available = self.engine.is_gpu_available()

            # Группа 1: Вычислительное устройство
            device_group = QGroupBox("Вычислительное устройство")
            device_group_layout = QVBoxLayout(device_group)
            device_group_layout.setSpacing(10)
            self.radio_cpu = QRadioButton("Использовать CPU (Центральный процессор)")
            self.radio_gpu = QRadioButton("Использовать GPU CUDA (Рекомендуется)")
            
            if gpu_available:
                self.radio_gpu.setChecked(True)
            else:
                self.radio_gpu.setEnabled(False)
                self.radio_gpu.setToolTip("CUDA-совместимая видеокарта не найдена или PyTorch не поддерживает её.")
                self.radio_cpu.setChecked(True)
                
            device_group_layout.addWidget(self.radio_gpu)
            device_group_layout.addWidget(self.radio_cpu)
            tab2_layout.addWidget(device_group)

            # Группа 2: Режимы точности TotalSegmentator
            precision_group = QGroupBox("Точность и разрешение ИИ")
            precision_group_layout = QVBoxLayout(precision_group)
            
            self.precision_combo = QComboBox()
            self.precision_combo.addItems([
                "Стандартная (1.5 мм разрешение, стандарт)",
                "Быстрая (3.0 мм разрешение, быстро)",
                "Ультра-быстрая (Body - поиск контура тела целиком)"
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
            
            self.clean_blobs_check = QCheckBox("Remove small blobs (Удалять мелкие артефакты)")
            self.clean_blobs_check.setToolTip(
                "Удаляет изолированный мелкий шум нейросети на КТ-срезах,\n"
                "оставляя только основной объем органа."
            )
            self.clean_blobs_check.setChecked(True)
            
            self.smoothing_check = QCheckBox("Smoothing (Сглаживание контуров)")
            self.smoothing_check.setToolTip(
                "Применяет Гауссову фильтрацию к 3D-маске, убирая «ступенчатость» срезов."
            )
            self.smoothing_check.stateChanged.connect(self.on_smoothing_check_changed)
            
            smoothing_param_layout = QHBoxLayout()
            smoothing_param_label = QLabel("Уровень сглаживания:")
            self.smoothing_combo = QComboBox()
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
            color_group = QGroupBox("Управление цветами ROI")
            color_group_layout = QVBoxLayout(color_group)
            
            color_preset_label = QLabel("Предопределенный набор цветов:")
            self.color_preset_combo = QComboBox()
            self.color_preset_combo.addItems([
                "Классический AI Contour",
                "QUANTEC",
                "Яркий неоновый"
            ])
            self.color_preset_combo.currentTextChanged.connect(self.on_color_preset_changed)
            
            color_group_layout.addWidget(color_preset_label)
            color_group_layout.addWidget(self.color_preset_combo)
            tab2_layout.addWidget(color_group)

            # Звук в конце
            self.sound_check = QCheckBox("Звуковое оповещение при завершении 🔔")
            self.sound_check.setChecked(True)
            tab2_layout.addWidget(self.sound_check)
            
            tab2_layout.addStretch()

            self.tab_widget.addTab(tab2_widget, "⚙️ Настройки")

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
    <h1>Справка по работе с AI Contour 📖</h1>

    <p><b>AI Contour</b> — интеллектуальное ПО для автоматического сегментирования органов риска (OAR) на КТ-снимках DICOM с использованием нейросети <b>TotalSegmentator</b>.</p>

    <div class="card">
        <h2>Основные возможности 🚀</h2>
        <ul>
            <li><b>Динамические пресеты:</b> Редактируйте анатомические пресеты во внешнем файле <span class="highlight">presets.json</span>.</li>
            <li><b>GPU-ускорение:</b> При наличии Nvidia CUDA расчёты выполняются в 20–30 раз быстрее.</li>
            <li><b>3D постобработка:</b> Очистка мелкого шума (Remove small blobs) и сглаживание Гаусса.</li>
            <li><b>Кастомизация цветов:</b> Двойной клик по органу — выбор цвета. Палитры QUANTEC и Неон.</li>
            <li><b>Просмотр структур:</b> Включите «Отображать структуры» на вкладке снимков для наложения контуров на КТ.</li>
            <li><b>Режим слияния:</b> Дополняйте существующий RTSTRUCT или создавайте новый.</li>
        </ul>
    </div>

    <div class="card">
        <h2>Порядок работы 📋</h2>
        <ul>
            <li>На вкладке <b>«⚙️ Настройки»</b> выберите папку с КТ-снимками DICOM и нажмите <b>«📂 Источник»</b>.</li>
            <li>На вкладке <b>«🎯 Контуры и снимки»</b> выберите пресет органов и пациента в таблице.</li>
            <li>Настройте режим расчёта (CPU/GPU), точность ИИ и постобработку.</li>
            <li>Нажмите <b>«ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ»</b> и дождитесь завершения.</li>
            <li>После завершения включите <b>«Отображать структуры»</b> для просмотра результатов.</li>
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

            splitter.addWidget(left_card)

            # --- ПРАВАЯ КОЛОНКА (Терминал логов и управление) ---
            right_card = QFrame()
            right_card.setObjectName("card")
            right_layout = QVBoxLayout(right_card)
            right_layout.setSpacing(12)

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

            self.btn_run = QPushButton("ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ")
            self.btn_run.setObjectName("btnRun")
            self.btn_run.clicked.connect(self.start_segmentation)

            # Вертикальный сплиттер для главной зоны и зоны логов
            v_splitter = QSplitter(Qt.Orientation.Vertical)
            
            # Верхняя панель (Таблица + Вьюер)
            top_panel = QWidget()
            top_layout = QVBoxLayout(top_panel)
            top_layout.setContentsMargins(0, 0, 0, 0)
            
            top_layout.addWidget(table_header)
            
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
            
            self.chk_show_structures = QCheckBox("Отображать структуры")
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
            self.chk_show_structures.stateChanged.connect(self.on_show_structures_changed)
            
            self.rtstruct_combo = QComboBox()
            self.rtstruct_combo.setEnabled(False)
            self.rtstruct_combo.currentIndexChanged.connect(self.on_show_structures_changed)
            
            viewer_tools_layout.addWidget(self.chk_show_structures)
            viewer_tools_layout.addWidget(QLabel("Файл:"))
            viewer_tools_layout.addWidget(self.rtstruct_combo, 1)
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
            main_splitter = QSplitter(Qt.Orientation.Horizontal)
            main_splitter.addWidget(self.series_table)
            main_splitter.addWidget(viewer_container)
            main_splitter.setStretchFactor(0, 6)
            main_splitter.setStretchFactor(1, 4)
            main_splitter.setSizes([600, 400])
            
            top_layout.addWidget(main_splitter, 1)  # stretch=1: main_splitter занимает всё оставшееся пространство
            
            # Нижняя панель (Логи + Прогресс)
            bottom_panel = QWidget()
            bottom_layout = QVBoxLayout(bottom_panel)
            bottom_layout.setContentsMargins(0, 0, 0, 0)
            bottom_layout.setSpacing(6)
            
            bottom_layout.addWidget(logs_header)
            bottom_layout.addWidget(self.log_edit, 1)  # лог растягивается
            bottom_layout.addWidget(progress_header)
            bottom_layout.addWidget(self.status_step_label)
            bottom_layout.addWidget(self.progress_bar)
            bottom_layout.addWidget(self.eta_label)
            bottom_layout.addWidget(self.btn_run)
            
            v_splitter.addWidget(top_panel)
            v_splitter.addWidget(bottom_panel)
            # Вертикальный сплиттер: 50% верхняя зона / 50% логи
            v_splitter.setStretchFactor(0, 1)
            v_splitter.setStretchFactor(1, 1)
            v_splitter.setSizes([500, 500])
            
            right_layout.addWidget(v_splitter, 1)  # stretch=1: v_splitter заполняет right_card

            splitter.addWidget(right_card)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)

            # Псевдонимы для совместимости с требованиями ТЗ
            self.palette_combo = self.color_preset_combo
            self.structures_list = self.organs_list

            # Инициализация списков пресетов и органов из presets.json движка
            self.init_presets_and_organs()

            # Установка палитры по умолчанию на QUANTEC
            self.palette_combo.setCurrentText("QUANTEC")

            # Подключаем сохранение настроек
            self.sound_check.stateChanged.connect(self.on_sound_check_changed)
            self.clean_blobs_check.stateChanged.connect(self.save_settings)
            self.smoothing_check.stateChanged.connect(self.save_settings)
            self.precision_combo.currentIndexChanged.connect(self.save_settings)
            self.smoothing_combo.currentIndexChanged.connect(self.save_settings)
            self.color_preset_combo.currentIndexChanged.connect(self.save_settings)
            
            splitter.setSizes([430, 490])

        def on_sound_check_changed(self):
            self.save_settings()
            if self.sound_check.isChecked():
                try:
                    import winsound
                    winsound.Beep(523, 150)
                except Exception:
                    pass

        def init_presets_and_organs(self):
            """Инициализирует комбобокс пресетов и список органов из presets.json."""
            self.is_updating_presets = True
            self.preset_combo.clear()
            self.organs_list.clear()

            # Первый элемент — пустая строка-подсказка (ничего не выделяет)
            self.preset_combo.addItem("— Выберите пресет —")
            # Добавляем пресеты из движка
            presets_keys = list(self.engine.presets.keys())
            self.preset_combo.addItems(presets_keys)
            self.preset_combo.addItem("Все органы (All)")
            self.preset_combo.addItem("Пользовательский (Custom)")

            # Использование глобального ORGAN_GROUPS из config.py

            # Получаем все доступные органы динамически из движка
            all_supported_organs = self.engine.get_all_supported_organs()
            placed_organs = set()

            for group_title, organs in ORGAN_GROUPS.items():
                header_item = QListWidgetItem(f"{group_title} ({len(organs)})")
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
                    item = QListWidgetItem(f"   {ru_name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    
                    # Установка цветного квадратика-иконки для OAR
                    self.update_item_color_icon(item, org)
                    
                    self.organs_list.addItem(item)
            
            # Исключаем дубликаты с похожими названиями, чтобы они не засоряли раздел "Остальное"
            duplicates_to_exclude = {
                "brainstem", "eye_lens_left", "eye_lens_right", "iliac_vena_left", "iliac_vena_right",
                "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
                "kidney_cyst_left", "kidney_cyst_right", "thalamus", "caudate_nucleus", "lentiform_nucleus", "ventricle",
                "heart_myocardium", "heart_atrium_left", "heart_atrium_right", "heart_ventricle_left", "heart_ventricle_right"
            }
            other_organs = [org for org in all_supported_organs if org not in placed_organs and org not in duplicates_to_exclude]
            
            if other_organs:
                other_header = QListWidgetItem(f"━━━ ОСТАЛЬНОЕ ━━━ ({len(other_organs)})")
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
                    item = QListWidgetItem(f"   {ru_name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    self.update_item_color_icon(item, org)
                    self.organs_list.addItem(item)

            self.is_updating_presets = False
            self.update_checked_organs_count()

        def update_item_color_icon(self, item: QListWidgetItem, organ_name: str):
            """Генерирует и устанавливает цветную иконку для органа в списке."""
            pixmap = QPixmap(14, 14)
            color_rgb = self.engine.colors.get(organ_name, [128, 128, 128])
            pixmap.fill(QColor(color_rgb[0], color_rgb[1], color_rgb[2]))
            item.setIcon(QIcon(pixmap))

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

                color_preset = self.settings.value("color_preset", "Классический AI Contour")
                self.color_preset_combo.setCurrentText(color_preset)

                play_sound = self.settings.value("play_sound", True, type=bool)
                self.sound_check.setChecked(play_sound)

                # Загружаем выбранные ресурсы
                use_gpu = self.settings.value("use_gpu", True, type=bool)
                if self.radio_gpu.isEnabled():
                    self.radio_gpu.setChecked(use_gpu)
                    self.radio_cpu.setChecked(not use_gpu)
                else:
                    self.radio_cpu.setChecked(True)

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
            target_text = custom_text if custom_text else ("ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ" if is_patient_selected else "ВЫБЕРИТЕ ПАЦИЕНТА В ТАБЛИЦЕ")
            target_enabled = is_patient_selected if custom_text != "КТ-СЕРИИ НЕ НАЙДЕНЫ" else False
            
            if self.btn_run.text() != target_text:
                self.btn_run.setText(target_text)
                
            if self.btn_run.isEnabled() != target_enabled:
                self.btn_run.setEnabled(target_enabled)
                
            current_style = self.btn_run.styleSheet()
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
                self.update_run_button(True, "ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ")
                row = selected[0].row()
                selected_path = self.series_table.item(row, 6).text()
                
                # Меняем UI и обновляем вьюер ТОЛЬКО при ручном клике пользователя
                if not getattr(self, "_is_updating_table", False):
                    # Фоновый поиск реального пути файла (теперь не моргает от таймера)
                    self.check_for_rtstruct(selected_path)
                    
                    self.update_viewer_with_dicom(selected_path)
                    
                    str_status = self.series_table.item(row, 2).text()
                    if str_status == "Нет" or str_status == "No":
                        self.status_rtstruct_label.setText("RTSTRUCT: не найден (будет создан новый)")
                        self.status_rtstruct_label.setStyleSheet("color: #888888;")
                        self.radio_merge_merge.setEnabled(False)
                        self.radio_merge_overwrite.setEnabled(False)
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
                        self.radio_merge_overwrite.setEnabled(True)

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

        def check_for_rtstruct(self, directory: str):
            """Находит все RTSTRUCT файлы в выбранной папке."""
            self.existing_rtstruct_path = None
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
                
                # Принудительно очищаем старый оверлей из вьюера
                self.on_show_structures_changed()
            
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
                self.existing_rtstruct_path = self.rtstruct_files[-1]
                if hasattr(self, 'rtstruct_combo'):
                    self.rtstruct_combo.blockSignals(True)
                    for f in self.rtstruct_files:
                        self.rtstruct_combo.addItem(os.path.basename(f), f)
                    self.rtstruct_combo.setCurrentIndex(len(self.rtstruct_files) - 1)
                    self.rtstruct_combo.setEnabled(True)
                    self.chk_show_structures.setEnabled(True)
                    self.rtstruct_combo.blockSignals(False)
                    if self.chk_show_structures.isChecked():
                        self.on_show_structures_changed()

        def on_show_structures_changed(self):
            import pyqtgraph as pg
            import numpy as np
            from PyQt6.QtWidgets import QApplication, QProgressDialog
            from PyQt6.QtCore import Qt
            
            # Удаляем старый оверлей
            if hasattr(self, 'roi_overlay_item'):
                if self.roi_overlay_item in self.dicom_viewer.getView().addedItems:
                    self.dicom_viewer.getView().removeItem(self.roi_overlay_item)
                del self.roi_overlay_item
                if hasattr(self, 'roi_overlay_3d'):
                    del self.roi_overlay_3d
                    
            if not getattr(self, 'current_dicom_dir', None) or getattr(self, 'volume_3d_base', None) is None:
                return
                
            if not getattr(self, 'chk_show_structures', None) or not self.chk_show_structures.isChecked():
                self._last_loaded_rtstruct = None
                return
                
            rtstruct_path = self.rtstruct_combo.currentData()
            if not rtstruct_path or not os.path.exists(rtstruct_path):
                return
            
            is_new_rtstruct = (getattr(self, "_last_loaded_rtstruct", None) != rtstruct_path)
            
            progress_dialog = None
            try:
                # Устанавливаем форму курсора WaitCursor и выводим красивый статус
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.status_step_label.setText("⏳ Подготовка 3D-сцены: чтение DICOM RTSTRUCT файла...")
                QApplication.processEvents()
                
                from rt_utils import RTStructBuilder
                
                rtstruct = RTStructBuilder.create_from(
                    dicom_series_path=self.current_dicom_dir,
                    rt_struct_path=rtstruct_path,
                    warn_only=True
                )
                roi_names = rtstruct.get_roi_names()
                total_rois = len(roi_names)
                
                # Создаем красивое модальное окно прогресса
                progress_dialog = QProgressDialog("⏳ Инициализация 3D-структур...", None, 0, total_rois, self)
                progress_dialog.setWindowTitle("Загрузка 3D-контуров")
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.setCancelButton(None)
                progress_dialog.setStyleSheet(self.styleSheet())
                progress_dialog.setValue(0)
                progress_dialog.show()
                QApplication.processEvents()
                
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

                def get_mapped_organ(roi_name_str: str) -> str:
                    norm = normalize_name(roi_name_str)
                    if norm in EXTERNAL_ALIASES:
                        return EXTERNAL_ALIASES[norm]
                    if norm in supported_norm_map:
                        return supported_norm_map[norm]
                    return roi_name_str.lower().replace(" ", "_")

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

                # Собираем отмеченные органы для фильтрации вьюера
                checked_organs = set()
                for i in range(self.organs_list.count()):
                    itm = self.organs_list.item(i)
                    itm_data = itm.data(Qt.ItemDataRole.UserRole)
                    if itm_data != "header" and itm.checkState() == Qt.CheckState.Checked:
                        if isinstance(itm_data, dict):
                            org_name = itm_data.get("name") or (list(itm_data.keys())[0] if itm_data else "")
                        else:
                            org_name = itm_data
                        if org_name:
                            checked_organs.add(org_name.lower())
                
                # Если ни одна галочка не совпала с органами файла - показываем все (кроме body, который рисуется всегда)
                file_organs_no_body = file_organs - {"body"}
                if not checked_organs.intersection(file_organs_no_body):
                    checked_organs = file_organs_no_body
                
                z_dim, x_dim, y_dim = self.volume_3d_base.shape
                overlay_3d = np.zeros((z_dim, x_dim, y_dim, 4), dtype=np.uint8)
                
                for idx, roi in enumerate(roi_names, start=1):
                    try:
                        orig_organ = get_mapped_organ(roi)
                        
                        # Получаем красивое русское название органа для пошагового вывода
                        ru_name = self.engine.ru_names.get(orig_organ, orig_organ)
                        if orig_organ == "body":
                            ru_name = "Контур тела (Body)"
                        
                        # Обновляем прогресс
                        if progress_dialog:
                            progress_dialog.setValue(idx)
                            progress_dialog.setLabelText(f"⏳ Отрисовка контуров: {ru_name} ({idx}/{total_rois})...")
                        self.status_step_label.setText(f"⏳ Отрисовка контуров: {ru_name} ({idx}/{total_rois})...")
                        QApplication.processEvents()
                        
                        if orig_organ != "body" and orig_organ not in checked_organs:
                            continue
                        
                        # Используем безопасный slice-by-slice экстрактор вместо get_roi_mask_by_name,
                        # чтобы не падать при вырожденных контурах (Skull, Eye Right и др.)
                        mask_3d = self._get_roi_mask_safe(
                            rtstruct, roi,
                            self.volume_3d_base.shape,
                            getattr(self, 'z_positions', None),
                            getattr(self, 'dicom_pixel_spacing', (1.0, 1.0)),
                            getattr(self, 'dicom_image_position', [0.0, 0.0])
                        )   # возвращает (z, x, y) bool array
                        
                        if orig_organ == "body":
                            import scipy.ndimage
                            # Получаем тонкую линию силуэта кожи
                            boundary = mask_3d ^ scipy.ndimage.binary_erosion(mask_3d, structure=np.ones((1, 3, 3)))
                            # Закрашиваем светло-серым цветом [220, 220, 220] с высокой непрозрачностью 255
                            overlay_3d[boundary, 0] = 220
                            overlay_3d[boundary, 1] = 220
                            overlay_3d[boundary, 2] = 220
                            overlay_3d[boundary, 3] = 255
                        else:
                            color = self.engine.colors.get(orig_organ, [0, 255, 128])
                            overlay_3d[mask_3d, 0] = color[0]
                            overlay_3d[mask_3d, 1] = color[1]
                            overlay_3d[mask_3d, 2] = color[2]
                            overlay_3d[mask_3d, 3] = 100
                    except Exception as roi_e:
                        logger.warning(f"Не удалось отрисовать структуру {roi}: {roi_e}")
                
                self.roi_overlay_3d = overlay_3d
                self.roi_overlay_item = pg.ImageItem()
                self.roi_overlay_item.setZValue(10)
                self.dicom_viewer.getView().addItem(self.roi_overlay_item)
                
                self.update_roi_overlay_frame()
                self.status_step_label.setText("Текущий шаг: Ожидание запуска...")
            except Exception as e:
                logger.error(f"Ошибка загрузки структур во вьюер: {e}")
                self.status_step_label.setText("Текущий шаг: Ожидание запуска...")
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
            if hasattr(self, 'roi_overlay_item') and hasattr(self, 'roi_overlay_3d'):
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
            
            self.save_settings()
            self.update_checked_organs_count()

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
            
            self.save_settings()
            self.update_checked_organs_count()

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

            if preset_name == "Все органы (All)":
                target_organs = self.engine.get_all_supported_organs()
            else:
                target_organs_raw = self.engine.presets.get(preset_name, [])
                target_organs = []
                for item in target_organs_raw:
                    if isinstance(item, dict):
                        target_organs.extend(item.keys())
                    else:
                        target_organs.append(item)

            # Блокируем сигналы чтобы не вызывать on_organ_item_changed в цикле
            self.organs_list.blockSignals(True)
            try:
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
                "Классический AI Contour": {"spleen": [156, 39, 176], "kidney_right": [3, 169, 244], "kidney_left": [33, 150, 243], "gallbladder": [76, 175, 80], "liver": [139, 195, 74], "stomach": [255, 152, 0], "aorta": [244, 67, 54], "inferior_vena_cava": [63, 81, 181], "urinary_bladder": [255, 235, 59], "heart": [233, 30, 99], "lung_left": [0, 150, 136], "lung_right": [0, 188, 212], "trachea": [121, 85, 72], "esophagus": [158, 158, 158], "pancreas": [255, 193, 7], "duodenum": [173, 20, 87], "adrenal_gland_left": [255, 87, 34], "adrenal_gland_right": [255, 112, 67], "pulmonary_artery": [0, 150, 255], "small_bowel": [103, 58, 183], "prostate": [233, 30, 99], "rectum": [121, 85, 72], "colon": [0, 121, 107], "femur_left": [255, 224, 178], "femur_right": [255, 224, 178], "hip_left": [230, 238, 156], "hip_right": [230, 238, 156], "sacrum": [141, 110, 99], "spinal_cord": [0, 255, 0], "thyroid_gland": [255, 105, 180], "skull": [255, 228, 196], "brain": [135, 206, 250], "common_carotid_artery_left": [220, 20, 60], "common_carotid_artery_right": [220, 20, 60], "superior_vena_cava": [70, 130, 180], "portal_vein_and_splenic_vein": [0, 139, 139], "clavicula_left": [244, 164, 96], "clavicula_right": [244, 164, 96], "sternum": [222, 184, 135], "iliac_artery_left": [255, 99, 71], "iliac_artery_right": [255, 99, 71], "eye_left": [255, 255, 0], "eye_right": [255, 255, 0], "lens_left": [255, 165, 0], "lens_right": [255, 165, 0], "brain_stem": [210, 105, 30], "optic_nerve_left": [240, 230, 140], "optic_nerve_right": [240, 230, 140]},
                "QUANTEC": {"spleen": [160, 32, 240], "kidney_right": [0, 0, 255], "kidney_left": [30, 144, 255], "gallbladder": [0, 255, 0], "liver": [34, 139, 34], "stomach": [218, 165, 32], "aorta": [55, 197, 94], "inferior_vena_cava": [194, 166, 130], "urinary_bladder": [255, 215, 0], "heart": [255, 0, 0], "lung_left": [86, 123, 174], "lung_right": [195, 54, 110], "trachea": [149, 58, 171], "esophagus": [138, 127, 103], "pancreas": [153, 97, 184], "duodenum": [168, 85, 61], "adrenal_gland_left": [114, 125, 152], "adrenal_gland_right": [161, 157, 200], "pulmonary_artery": [98, 122, 139], "small_bowel": [177, 66, 127], "prostate": [152, 133, 118], "rectum": [139, 69, 19], "colon": [191, 68, 120], "femur_left": [135, 139, 183], "femur_right": [159, 155, 157], "hip_left": [146, 175, 165], "hip_right": [85, 193, 174], "sacrum": [96, 111, 190], "spinal_cord": [116, 98, 57], "thyroid_gland": [113, 52, 117], "skull": [94, 188, 72], "brain": [155, 169, 192], "common_carotid_artery_left": [51, 115, 144], "common_carotid_artery_right": [86, 147, 196], "superior_vena_cava": [84, 137, 160], "portal_vein_and_splenic_vein": [113, 127, 112], "clavicula_left": [144, 51, 84], "clavicula_right": [176, 73, 124], "sternum": [85, 68, 152], "iliac_artery_left": [134, 69, 129], "iliac_artery_right": [78, 137, 190], "eye_left": [255, 255, 100], "eye_right": [255, 255, 100], "lens_left": [255, 140, 0], "lens_right": [255, 140, 0], "brain_stem": [139, 69, 19], "optic_nerve_left": [255, 215, 0], "optic_nerve_right": [255, 215, 0]},
                "Яркий неоновый": {"spleen": [255, 0, 255], "kidney_right": [0, 255, 255], "kidney_left": [0, 191, 255], "gallbladder": [50, 205, 50], "liver": [173, 255, 47], "stomach": [255, 165, 0], "aorta": [255, 255, 0], "inferior_vena_cava": [128, 0, 255], "urinary_bladder": [255, 255, 0], "heart": [255, 20, 147], "lung_left": [255, 0, 255], "lung_right": [255, 0, 255], "trachea": [128, 255, 0], "esophagus": [0, 128, 255], "pancreas": [0, 128, 255], "duodenum": [255, 255, 0], "adrenal_gland_left": [255, 255, 0], "adrenal_gland_right": [255, 0, 128], "pulmonary_artery": [0, 128, 255], "small_bowel": [0, 0, 255], "prostate": [255, 0, 0], "rectum": [210, 105, 30], "colon": [0, 128, 255], "femur_left": [0, 255, 128], "femur_right": [128, 255, 0], "hip_left": [128, 0, 255], "hip_right": [0, 0, 255], "sacrum": [255, 0, 255], "spinal_cord": [255, 0, 128], "thyroid_gland": [0, 0, 255], "skull": [255, 0, 0], "brain": [0, 0, 255], "common_carotid_artery_left": [0, 255, 0], "common_carotid_artery_right": [0, 0, 255], "superior_vena_cava": [0, 128, 255], "portal_vein_and_splenic_vein": [0, 255, 0], "clavicula_left": [0, 0, 255], "clavicula_right": [255, 0, 128], "sternum": [0, 255, 0], "iliac_artery_left": [128, 0, 255], "iliac_artery_right": [128, 0, 255], "eye_left": [255, 255, 0], "eye_right": [255, 255, 0], "lens_left": [255, 69, 0], "lens_right": [255, 69, 0], "brain_stem": [255, 105, 180], "optic_nerve_left": [255, 215, 0], "optic_nerve_right": [255, 215, 0]}
            }

            palette = preset_palettes.get(text)
            if palette:
                for organ, color in palette.items():
                    if organ in self.engine.colors:
                        self.engine.colors[organ] = color
                
                # Сохраняем в presets.json
                self.engine.save_presets_config()
                
                # Обновляем все иконки в списке с временной блокировкой сигналов
                self.structures_list.blockSignals(True)
                try:
                    for i in range(self.structures_list.count()):
                        itm = self.structures_list.item(i)
                        org = itm.data(Qt.ItemDataRole.UserRole)
                        if org != "header":
                            self.update_item_color_icon(itm, org)
                finally:
                    self.structures_list.blockSignals(False)
                
                logger.info(f"Цветовая гамма переключена на пресет: '{text}'")

        def on_smoothing_check_changed(self, state: int):
            """Слот изменения состояния чекбокса сглаживания."""
            enabled = (state == 2)
            self.smoothing_combo.setEnabled(enabled)

        def append_log(self, message: str, color: str):
            """Потокобезопасное добавление логов в текстовое окно."""
            self.log_edit.append(f"<span style='color: {color};'>{message}</span>")
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
            elif self.radio_merge_overwrite.isChecked():
                merge_mode = "overwrite"
            else:
                merge_mode = "merge"
                
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

                # Создаем и запускаем поток вычислений
                self.worker = SegmentationWorker(
                    engine=self.engine,
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
                
                self.current_step_base_text = "Подготовка пайплайна..."
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
            self.btn_select_all.setEnabled(enabled)
            self.btn_deselect_all.setEnabled(enabled)
            self.preset_combo.setEnabled(enabled)
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
            self.radio_gpu.setEnabled(enabled and self.engine.is_gpu_available())
            
            # Блокировка/восстановление переключателей слияния RTSTRUCT
            if enabled:
                row = self.series_table.currentRow()
                if row >= 0:
                    str_status = self.series_table.item(row, 2).text()
                    has_structs = (str_status != "Нет" and str_status != "No" and "0 " not in str_status)
                    self.radio_merge_new.setEnabled(True)
                    self.radio_merge_merge.setEnabled(has_structs)
                    self.radio_merge_overwrite.setEnabled(has_structs)
                else:
                    self.radio_merge_new.setEnabled(True)
                    self.radio_merge_merge.setEnabled(False)
                    self.radio_merge_overwrite.setEnabled(False)
            else:
                self.radio_merge_new.setEnabled(False)
                self.radio_merge_merge.setEnabled(False)
                self.radio_merge_overwrite.setEnabled(False)
            
            self.btn_run.setEnabled(True)
            if enabled:
                self.btn_run.setText("ЗАПУСТИТЬ АВТООКОНТУРИРОВАНИЕ")
                self.btn_run.setStyleSheet("")
            else:
                self.btn_run.setText("Отменить автооконтуривание")
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
            <li><b>Динамические пресеты:</b> Вы можете легко добавлять или редактировать анатомические пресеты во внешнем файле <span class="highlight">presets.json</span> в корневой папке проекта.</li>
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
                event.accept()
            else:
                event.ignore()


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

    # Защита от запуска второй копии программы (используем мьютекс Windows)
    try:
        import ctypes
        mutex_name = "AIContourAppMutex_1.0"
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
