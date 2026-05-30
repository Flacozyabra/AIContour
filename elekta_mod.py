#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Модуль elekta_mod.py: Реализация неблокирующего режима 'Elekta mod'
================================================================================
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from pydicom import dcmread
from pynetdicom import AE, evt, StoragePresentationContexts

# Настройка логирования
logger = logging.getLogger("ElektaMod")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s]: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class DicomReceiver(QObject):
    """
    Неблокирующий DICOM C-STORE SCP сервер приема (на базе QObject с block=False).
    Позволяет мгновенно запускать и безопасно останавливать сервер, исключая зависания GUI.
    """
    study_received = pyqtSignal(str, str)  # Сигнал (путь_к_папке, patient_id)
    file_received = pyqtSignal(str)        # Сигнал прогресса для логов
    error_occurred = pyqtSignal(str)       # Сигнал об ошибке

    def __init__(self, port: int = 10404, ae_title: str = "AIC_SCP", output_dir: str = "DICOM") -> None:
        super().__init__()
        self.port = port
        self.ae_title = ae_title
        self.output_dir = output_dir
        self.server = None
        self.last_sender_ip: Optional[str] = None
        self.received_files: List[str] = []
        self.current_patient_id: Optional[str] = None
        self.current_study_dir: Optional[str] = None

    def start(self) -> bool:
        """Запускает DICOM C-STORE SCP сервер в неблокирующем фоновом потоке pynetdicom."""
        self.stop()
        self.received_files = []
        self.current_patient_id = None
        self.current_study_dir = None
        
        ae = AE(ae_title=self.ae_title.encode('utf-8'))
        ae.supported_contexts = StoragePresentationContexts

        def handle_store(event) -> int:
            try:
                # Извлекаем IP-адрес отправителя
                self.last_sender_ip = event.assoc.requestor.address
                
                ds = event.dataset
                patient_id = str(getattr(ds, 'PatientID', 'UNKNOWN')).strip()
                patient_name = str(getattr(ds, 'PatientName', 'UNKNOWN')).replace('^', ' ').strip()
                study_date = str(getattr(ds, 'StudyDate', 'NODATE')).strip()
                
                dir_name = f"{patient_name}_{study_date}"
                dir_name = re.sub(r'[^a-zA-Z0-9\s_\-]', '', dir_name).strip()
                if not dir_name:
                    dir_name = "UnknownStudy"

                study_path = Path(self.output_dir) / dir_name
                study_path.mkdir(parents=True, exist_ok=True)
                
                self.current_study_dir = str(study_path)
                self.current_patient_id = patient_id

                sop_instance_uid = getattr(ds, 'SOPInstanceUID', 'file')
                file_path = study_path / f"{sop_instance_uid}.dcm"

                # Сохраняем файл на диск
                ds.save_as(str(file_path), write_like_original=False)
                
                self.received_files.append(str(file_path))
                self.file_received.emit(f"Принят файл DICOM: {patient_name} (ID: {patient_id}) -> {sop_instance_uid[:12]}...")
                
                return 0x0000  # Success
            except Exception as e:
                logger.error(f"Ошибка сохранения входящего DICOM-файла: {e}", exc_info=True)
                return 0xC000  # Cannot Understand

        def handle_release(event) -> None:
            if self.current_study_dir and self.received_files:
                logger.info(f"Ассоциация закрыта. Передача серии {self.current_study_dir} окончена. Файлов: {len(self.received_files)}.")
                self.study_received.emit(self.current_study_dir, self.current_patient_id or "")
                self.received_files = []

        handlers = [
            (evt.EVT_C_STORE, handle_store),
            (evt.EVT_RELEASED, handle_release)
        ]

        try:
            logger.info(f"Запуск неблокирующего C-STORE SCP сервера на порту {self.port} (AET: {self.ae_title})...")
            # block=False запускает сервер во внутреннем неблокирующем потоке pynetdicom
            self.server = ae.start_server(('', self.port), block=False, evt_handlers=handlers)
            return True
        except Exception as e:
            error_msg = f"Не удалось запустить DICOM C-STORE SCP сервер: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def stop(self) -> None:
        """Останавливает сервер мгновенно и безопасно."""
        if self.server:
            logger.info("Остановка неблокирующего C-STORE SCP сервера...")
            try:
                self.server.shutdown()
            except Exception as e:
                logger.error(f"Ошибка при выключении DICOM сервера: {e}")
            self.server = None


class DicomSenderThread(QThread):
    """
    Фоновый поток для отправки КТ-исследования и файлов RTSTRUCT обратно на Monaco (C-STORE SCU).
    """
    progress_signal = pyqtSignal(int, int, str)  # (текущий, всего, имя_файла)
    finished_signal = pyqtSignal(bool, str)       # (успех, сообщение_результата)

    def __init__(self, ip: str, port: int = 104, ae_title: str = "MONACO", 
                 local_ae_title: str = "AICONTOUR_SCU", files_to_send: List[str] = None) -> None:
        super().__init__()
        self.ip = ip
        self.port = port
        self.ae_title = ae_title
        self.local_ae_title = local_ae_title
        self.files_to_send = files_to_send or []

    def run(self) -> None:
        if not self.files_to_send:
            self.finished_signal.emit(False, "Список файлов для отправки обратно на Monaco пуст.")
            return

        ae = AE(ae_title=self.local_ae_title.encode('utf-8'))
        ae.requested_contexts = StoragePresentationContexts

        logger.info(f"Установка DICOM-ассоциации с Monaco ({self.ip}:{self.port}, AET: {self.ae_title})...")
        assoc = ae.associate(self.ip, self.port, ae_title=self.ae_title.encode('utf-8'))

        if assoc.is_established:
            total_files = len(self.files_to_send)
            success_count = 0
            
            for idx, file_path in enumerate(self.files_to_send):
                if not os.path.exists(file_path):
                    continue
                
                try:
                    ds = dcmread(file_path)
                    filename = os.path.basename(file_path)
                    
                    self.progress_signal.emit(idx + 1, total_files, filename)
                    
                    status = assoc.send_c_store(ds)
                    
                    if status and getattr(status, 'Status', None) == 0x0000:
                        success_count += 1
                    else:
                        logger.warning(f"Монако отверг C-STORE для файла {filename}. Статус: {status}")
                except Exception as e:
                    logger.error(f"Сбой при отправке файла {file_path}: {e}")

            assoc.release()
            
            if success_count == total_files:
                self.finished_signal.emit(True, f"Успешно экспортировано {success_count} файлов обратно на Monaco! ✅")
            else:
                self.finished_signal.emit(False, f"Экспорт завершен частично: отправлено {success_count} из {total_files} файлов. Проверьте логи.")
        else:
            self.finished_signal.emit(False, f"Не удалось подключиться к Monaco SCP ({self.ip}:{self.port}, AET: {self.ae_title}).")


class ElektaManager:
    """
    Класс-координатор для управления режимом Elekta mod.
    """
    def __init__(self, output_dir: str = "DICOM", log_callback: Optional[Callable[[str], None]] = None) -> None:
        self.output_dir = output_dir
        self.log_callback = log_callback
        self.receiver: Optional[DicomReceiver] = None
        self.sender_thread: Optional[DicomSenderThread] = None
        
        self.last_monaco_ip: Optional[str] = None
        self.monaco_port: int = 104
        self.monaco_aet: str = "MONACO"

    def _log(self, text: str) -> None:
        logger.info(text)
        if self.log_callback:
            self.log_callback(text)

    def start_receiver(self, port: int = 10404, ae_title: str = "AIC_SCP", 
                       study_received_callback: Optional[Callable[[str, str], None]] = None) -> bool:
        """Запускает неблокирующий приемник DICOM на указанном порту."""
        self.stop_receiver()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Автоматическая очистка исследований старше 1 дня перед запуском
        self.cleanup_old_studies(days=1.0)

        self.receiver = DicomReceiver(port=port, ae_title=ae_title, output_dir=self.output_dir)
        
        self.receiver.file_received.connect(self._log)
        self.receiver.error_occurred.connect(self._log)
        
        if study_received_callback:
            def on_study_received(path, patient_id):
                if self.receiver:
                    self.last_monaco_ip = self.receiver.last_sender_ip
                    self._log(f"Запомнен IP-адрес станции Monaco: {self.last_monaco_ip}")
                study_received_callback(path, patient_id)
            self.receiver.study_received.connect(on_study_received)
        else:
            def save_ip_only(path, patient_id):
                if self.receiver:
                    self.last_monaco_ip = self.receiver.last_sender_ip
            self.receiver.study_received.connect(save_ip_only)

        success = self.receiver.start()
        if success:
            self._log(f"Режим Elekta: запущен приемник DICOM на порту {port} (AET: {ae_title}). Ожидание данных...")
        return success

    def stop_receiver(self) -> None:
        """Останавливает приемник DICOM мгновенно."""
        if self.receiver:
            self._log("Режим Elekta: остановка приемника DICOM...")
            self.receiver.stop()
            self.receiver = None

    def send_back_to_monaco(self, study_dir: str, progress_callback: Optional[Callable[[int, int, str], None]] = None,
                            finished_callback: Optional[Callable[[bool, str], None]] = None) -> bool:
        """Запускает фоновую отправку всех файлов исследования и RTSTRUCT обратно на Monaco."""
        if not self.last_monaco_ip:
            self._log("Ошибка: нет сохраненного IP-адреса Monaco для отправки.")
            if finished_callback:
                finished_callback(False, "Отсутствует сохраненный IP-адрес Monaco (исследование не принималось по сети).")
            return False

        if not os.path.isdir(study_dir):
            self._log(f"Ошибка: директория исследования не найдена: {study_dir}")
            if finished_callback:
                finished_callback(False, f"Папка исследования не существует: {study_dir}")
            return False

        files_to_send = []
        for root, _, files in os.walk(study_dir):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or file.startswith('STR_'):
                    files_to_send.append(os.path.join(root, file))

        if not files_to_send:
            self._log(f"Ошибка: в папке {study_dir} не найдено DICOM-файлов.")
            if finished_callback:
                finished_callback(False, "В папке исследования не обнаружено DICOM файлов.")
            return False

        self._log(f"Запуск экспорта обратно на Monaco ({self.last_monaco_ip}:{self.monaco_port}, AET: {self.monaco_aet})...")
        self.sender_thread = DicomSenderThread(
            ip=self.last_monaco_ip,
            port=self.monaco_port,
            ae_title=self.monaco_aet,
            files_to_send=files_to_send
        )

        if progress_callback:
            self.sender_thread.progress_signal.connect(progress_callback)
        else:
            self.sender_thread.progress_signal.connect(lambda cur, tot, name: self._log(f"Экспорт [{cur}/{tot}]: {name}"))

        if finished_callback:
            self.sender_thread.finished_signal.connect(finished_callback)
        else:
            self.sender_thread.finished_signal.connect(lambda success, msg: self._log(msg))

        self.sender_thread.start()
        return True

    def cleanup_old_studies(self, days: float = 1.0) -> None:
        """
        Удаляет старые папки исследований из output_dir, которые старше указанного количества дней.
        """
        import time
        import shutil

        if not os.path.exists(self.output_dir):
            return

        now = time.time()
        cutoff = now - (days * 24 * 3600)
        deleted_count = 0

        try:
            for entry in os.scandir(self.output_dir):
                if entry.is_dir():
                    # Проверяем mtime самой директории
                    dir_mtime = entry.stat().st_mtime
                    
                    # Проверяем mtime файлов внутри на случай, если папка старая, но файлы свежие
                    try:
                        for subentry in os.scandir(entry.path):
                            if subentry.is_file():
                                dir_mtime = max(dir_mtime, subentry.stat().st_mtime)
                    except Exception:
                        pass

                    if dir_mtime < cutoff:
                        logger.info(f"Удаление старого исследования: {entry.path} (возраст: {(now - dir_mtime) / 3600:.1f} ч)")
                        shutil.rmtree(entry.path, ignore_errors=True)
                        deleted_count += 1

            if deleted_count > 0:
                self._log(f"Автоочистка DICOM: удалено {deleted_count} исследований старше {days} дн.")
        except Exception as e:
            logger.error(f"Ошибка при очистке старых исследований в {self.output_dir}: {e}", exc_info=True)
