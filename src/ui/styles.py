#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Модуль styles.py: Единые QSS стили оформления интерфейса AI Contour
================================================================================
"""

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
    background-color: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0.00 #00bda6, 
        stop: 0.25 #00bda6,
        stop: 0.25 #00a38f, 
        stop: 0.50 #00a38f,
        stop: 0.50 #00bda6, 
        stop: 0.75 #00bda6,
        stop: 0.75 #00a38f, 
        stop: 1.00 #00a38f
    );
    border-radius: 4px;
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

QPushButton#btnExitPreview {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #d87a00, stop: 1 #b76500);
    border: 1px solid #d87a00;
    font-size: 14px;
    font-weight: bold;
    padding: 12px;
    border-radius: 6px;
    color: #ffffff;
}

QPushButton#btnExitPreview:hover {
    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f39c12, stop: 1 #d87a00);
    border: 1px solid #f39c12;
}

QPushButton#btnExitPreview:pressed {
    background-color: #9e5100;
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

class StyleManager:
    """Менеджер стилизации QSS для приложения AI Contour."""
    @staticmethod
    def get_dark_theme() -> str:
        """Возвращает премиальную темную тему оформления в формате QSS."""
        return DARK_QSS
