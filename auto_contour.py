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
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Импорт PyQt6
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QComboBox, QListWidget, QListWidgetItem,
        QRadioButton, QButtonGroup, QTextEdit, QProgressBar, QFileDialog,
        QMessageBox, QFrame, QSplitter, QCheckBox, QDialog, QTextBrowser,
        QTabWidget, QColorDialog, QGroupBox
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

    class SegmentationWorker(QThread):
        """Поток для вычислений сегментации TotalSegmentator, чтобы GUI не зависал."""
        finished_signal = pyqtSignal(bool, str)
        step_signal = pyqtSignal(str)

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
                def callback(step_text: str):
                    self.step_signal.emit(step_text)
                    
                def reg_proc(p):
                    self.process = p
                    
                def is_canc():
                    return self.is_cancelled

                self.engine.run_pipeline(
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
                    is_cancelled_cb=is_canc,
                    register_process_cb=reg_proc
                )
                if self.is_cancelled:
                    self.finished_signal.emit(False, "Операция отменена пользователем.")
                else:
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

            # Определение GPU/CPU для подзаголовка
            gpu_available = self.engine.is_gpu_available()
            device_str = "CUDA GPU доступна" if gpu_available else "Доступен только CPU"

            # Шапка
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 5)

            title_layout = QVBoxLayout()
            title_layout.setSpacing(2)
            title = QLabel("AI Contour")
            title.setObjectName("titleLabel")
            self.subtitle_label = QLabel(f"Автоматическое сегментирование органов риска на КТ ({device_str})")
            self.subtitle_label.setObjectName("subtitleLabel")
            if gpu_available:
                self.subtitle_label.setStyleSheet("color: #2ecc71;")
            title_layout.addWidget(title)
            title_layout.addWidget(self.subtitle_label)

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

            # Выбор КТ DICOM
            input_label = QLabel("Папка с КТ-снимками DICOM:")
            input_label.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.input_edit = QLineEdit()
            self.input_edit.setPlaceholderText("Выберите папку с DICOM файлами...")
            self.input_edit.textChanged.connect(self.check_for_rtstruct)
            self.btn_input = QPushButton("📂 Обзор...")
            self.btn_input.setObjectName("btnBrowse")
            self.btn_input.clicked.connect(self.select_input_dir)

            input_box = QHBoxLayout()
            input_box.addWidget(self.input_edit)
            input_box.addWidget(self.btn_input)
            tab1_layout.addWidget(input_label)
            tab1_layout.addLayout(input_box)

            # Под-карточка статуса RTSTRUCT
            status_frame = QFrame()
            status_frame.setObjectName("statusCard")
            status_layout = QVBoxLayout(status_frame)
            status_layout.setSpacing(6)

            status_title = QLabel("Работа с существующими контурами:")
            status_title.setStyleSheet("font-weight: bold; color: #b0b0b0;")
            self.status_rtstruct_label = QLabel("Статус: Путь не выбран")
            self.status_rtstruct_label.setStyleSheet("color: #888888;")
            self.status_rtstruct_label.setWordWrap(True)

            status_layout.addWidget(status_title)
            status_layout.addWidget(self.status_rtstruct_label)
            tab1_layout.addWidget(status_frame)

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
            organs_header = QLabel("Органы для автооконтурирования:")
            organs_header.setStyleSheet("font-weight: bold; color: #ffffff;")
            self.organs_list = QListWidget()
            self.organs_list.itemChanged.connect(self.on_organ_item_changed)
            self.organs_list.itemSelectionChanged.connect(self.on_organ_selection_changed)

            tab1_layout.addWidget(organs_header)
            tab1_layout.addWidget(self.organs_list)
            
            # Кнопка индивидуальной настройки цвета выделенного органа
            self.btn_color_pick = QPushButton("🎨 Выбрать индивидуальный цвет органа...")
            self.btn_color_pick.setObjectName("btnAction")
            self.btn_color_pick.setEnabled(False)
            self.btn_color_pick.clicked.connect(self.pick_organ_color)
            tab1_layout.addWidget(self.btn_color_pick)
            
            self.tab_widget.addTab(tab1_widget, "🎯 Контуры и снимки")

            # ------------------------------------------------------------------
            # ВКЛАДКА 2: Параметры ИИ и Цвета
            # ------------------------------------------------------------------
            tab2_widget = QWidget()
            tab2_layout = QVBoxLayout(tab2_widget)
            tab2_layout.setSpacing(12)

            # Группа 1: Вычислительное устройство
            device_group = QGroupBox("Вычислительное устройство")
            device_group_layout = QVBoxLayout(device_group)
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
                "Клинический QUANTEC",
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

            splitter.addWidget(left_card)

            # --- ПРАВАЯ КОЛОНКА (Терминал логов и управление) ---
            right_card = QFrame()
            right_card.setObjectName("card")
            right_layout = QVBoxLayout(right_card)
            right_layout.setSpacing(12)

            logs_header = QLabel("Лог выполнения работы движка в реальном времени:")
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

            # Инициализация списков пресетов и органов из presets.json движка
            self.init_presets_and_organs()

            # Подключаем сохранение настроек
            self.sound_check.stateChanged.connect(self.on_sound_check_changed)
        
        def on_sound_check_changed(self):
            self.save_settings()
            if self.sound_check.isChecked():
                try:
                    import winsound
                    winsound.Beep(523, 150)
                except Exception:
                    pass
            self.clean_blobs_check.stateChanged.connect(self.save_settings)
            self.smoothing_check.stateChanged.connect(self.save_settings)
            self.precision_combo.currentIndexChanged.connect(self.save_settings)
            self.smoothing_combo.currentIndexChanged.connect(self.save_settings)
            self.color_preset_combo.currentIndexChanged.connect(self.save_settings)
            
            splitter.setSizes([430, 490])

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

            # Группировка списка органов по анатомическим областям
            ORGAN_GROUPS = {
                "━━━ ОБЩЕЕ ━━━": [
                    "body"
                ],
                "━━━ ГОЛОВА И ШЕЯ ━━━": [
                    "brain", "spinal_cord", "thyroid_gland", "skull", "trachea", "esophagus",
                    "common_carotid_artery_left", "common_carotid_artery_right"
                ],
                "━━━ ГРУДНАЯ КЛЕТКА ━━━": [
                    "heart", "lung_left", "lung_right", "trachea", "esophagus", "aorta", "pulmonary_artery",
                    "superior_vena_cava", "sternum", "clavicula_left", "clavicula_right"
                ],
                "━━━ БРЮШНАЯ ПОЛОСТЬ ━━━": [
                    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "inferior_vena_cava", "pancreas", "duodenum", "adrenal_gland_left", "adrenal_gland_right", "portal_vein_and_splenic_vein"
                ],
                "━━━ МАЛЫЙ ТАЗ ━━━": [
                    "urinary_bladder", "prostate", "rectum", "colon", "small_bowel", "femur_left", "femur_right", "hip_left", "hip_right", "sacrum", "iliac_artery_left", "iliac_artery_right"
                ]
            }

            for group_title, organs in ORGAN_GROUPS.items():
                header_item = QListWidgetItem(group_title)
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
                    # Проверяем, есть ли такой орган в ru_names
                    ru_name = self.engine.ru_names.get(org, org)
                    item = QListWidgetItem(f"   {ru_name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setData(Qt.ItemDataRole.UserRole, org)
                    
                    # Установка цветного квадратика-иконки для OAR
                    self.update_item_color_icon(item, org)
                    
                    self.organs_list.addItem(item)

            self.is_updating_presets = False

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

        def check_for_rtstruct(self, directory: str):
            """Автоматически сканирует папку КТ на наличие существующего RTSTRUCT файла."""
            self.existing_rtstruct_path = None
            if not directory or not os.path.isdir(directory):
                self.status_rtstruct_label.setText("Статус: Путь не выбран или недействителен")
                self.status_rtstruct_label.setStyleSheet("color: #888888;")
                # removed radio_merge and radio_new disables
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
                            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                            if getattr(ds, "Modality", None) == "RTSTRUCT":
                                found_file = filepath
                                break
                        except Exception:
                            continue
                
                if found_file:
                    self.existing_rtstruct_path = found_file
                    basename = os.path.basename(found_file)
                    self.status_rtstruct_label.setText(f"Обнаружен существующий RTSTRUCT: {basename}")
                    self.status_rtstruct_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
                    self.radio_merge.setEnabled(True)
                    self.radio_new.setEnabled(True)
                    self.radio_merge.setChecked(True)
                else:
                    self.status_rtstruct_label.setText("Существующий RTSTRUCT не обнаружен (будет создан новый)")
                    self.status_rtstruct_label.setStyleSheet("color: #e74c3c;")
                    

            except Exception as e:
                self.status_rtstruct_label.setText(f"Ошибка при сканировании RTSTRUCT: {str(e)}")
                self.status_rtstruct_label.setStyleSheet("color: #e74c3c;")
                # error opening dir

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
                        if set(checked_organs) == set(porgans):
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
                target_organs = list(self.engine.ru_names.keys())
            else:
                target_organs = self.engine.presets.get(preset_name, [])

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

            organ_name = item.data(Qt.ItemDataRole.UserRole)
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
                    if itm != item and itm.data(Qt.ItemDataRole.UserRole) == organ_name:
                        itm.setCheckState(state)
                self.is_updating_presets = False

            # Обновляем состояния всех заголовков групп
            self.update_headers_check_states()
            # Синхронизируем текст в комбобоксе пресетов
            self._sync_preset_combo_to_organs()
            self.save_settings()

        def on_organ_selection_changed(self):
            """Слот изменения выделенной строки в списке."""
            selected = self.organs_list.selectedItems()
            if not selected:
                self.btn_color_pick.setEnabled(False)
                return
            
            organ_name = selected[0].data(Qt.ItemDataRole.UserRole)
            if organ_name == "header":
                self.btn_color_pick.setEnabled(False)
            else:
                self.btn_color_pick.setEnabled(True)

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

        def on_color_preset_changed(self, text: str):
            """Слот изменения цветового пресета."""
            # Наборы пресетов
            preset_palettes = {
                "Классический AI Contour": {
                    "spleen": [156, 39, 176], "kidney_right": [3, 169, 244], "kidney_left": [33, 150, 243],
                    "gallbladder": [76, 175, 80], "liver": [139, 195, 74], "stomach": [255, 152, 0],
                    "urinary_bladder": [255, 235, 59], "heart": [233, 30, 99], "rectum": [121, 85, 72]
                },
                "Клинический QUANTEC": {
                    "spleen": [160, 32, 240], "kidney_right": [0, 0, 255], "kidney_left": [30, 144, 255],
                    "gallbladder": [0, 255, 0], "liver": [34, 139, 34], "stomach": [218, 165, 32],
                    "urinary_bladder": [255, 215, 0], "heart": [255, 0, 0], "rectum": [139, 69, 19]
                },
                "Яркий неоновый": {
                    "spleen": [255, 0, 255], "kidney_right": [0, 255, 255], "kidney_left": [0, 191, 255],
                    "gallbladder": [50, 205, 50], "liver": [173, 255, 47], "stomach": [255, 165, 0],
                    "urinary_bladder": [255, 255, 0], "heart": [255, 20, 147], "rectum": [210, 105, 30]
                }
            }

            palette = preset_palettes.get(text)
            if palette:
                for organ, color in palette.items():
                    if organ in self.engine.colors:
                        self.engine.colors[organ] = color
                
                # Сохраняем в presets.json
                self.engine.save_presets_config()
                
                # Обновляем все иконки в списке
                for i in range(self.organs_list.count()):
                    itm = self.organs_list.item(i)
                    org = itm.data(Qt.ItemDataRole.UserRole)
                    if org != "header":
                        self.update_item_color_icon(itm, org)
                
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
                return

            dicom_dir = self.input_edit.text().strip()
            if not dicom_dir or not os.path.isdir(dicom_dir):
                QMessageBox.critical(self, "Ошибка", "Укажите корректный путь к папке с КТ DICOM снимками!")
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
                org = item.data(Qt.ItemDataRole.UserRole)
                if org == "header":
                    continue
                if item.checkState() == Qt.CheckState.Checked:
                    if org not in selected_organs:
                        selected_organs.append(org)
                    
            if not selected_organs:
                QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одного органа для сегментирования!")
                return
                
            merge_mode = bool(self.existing_rtstruct_path)
            
            # Блокируем интерфейс
            self.set_ui_enabled(False)
            self.log_edit.clear()
            self.progress_bar.setValue(0)
            
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
            
            self.current_step_base_text = "Подготовка пайплайна..."
            self.spinner_index = 0
            self.pulse_tick = 0
            self.activity_timer.start()
            
            self.worker.start()

        def set_ui_enabled(self, enabled: bool):
            self.input_edit.setEnabled(enabled)
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
                self.btn_color_pick.setEnabled(False)
                
            self.color_preset_combo.setEnabled(enabled)
            # radio disables removed
            
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
            self.set_ui_enabled(True)
            self.progress_bar.setRange(0, 100)
            self.activity_timer.stop()
            self.status_step_label.setStyleSheet("color: #007acc; font-weight: bold; font-style: italic;")
            
            if self.sound_check.isChecked():
                try:
                    import winsound
                    import time
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
                QMessageBox.information(self, "Успех", "Автоматическое оконтурирование завершено успешно!")
            else:
                self.progress_bar.setValue(0)
                if "отменена пользователем" in message.lower() or "отменен пользователем" in message.lower():
                    self.status_step_label.setText("Текущий шаг: Расчет отменен!")
                    self.status_step_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                    QMessageBox.warning(self, "Предупреждение", "Процесс оконтурирования был прерван.")
                else:
                    self.status_step_label.setText("Текущий шаг: Ошибка!")
                    QMessageBox.critical(self, "Критическая ошибка", f"Произошел сбой при сегментации:\n{message}")

        def on_step_changed(self, step_text: str):
            self.current_step_base_text = step_text
            self.status_step_label.setText(f"{step_text} {self.SPINNER_FRAMES[self.spinner_index]}")
            
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
            event.accept()


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

    # Иконка на QApplication охватывает все окна приложения
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
