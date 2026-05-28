#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Модуль server_api.py: FastAPI REST API для управления задачами оконтуривания
================================================================================
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# Импортируем менеджер очереди
from server.queue_manager import QueueManager

logger = logging.getLogger("ServerAPI")

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

class BlacklistManager:
    def __init__(self):
        self.active_clients = {}  # client_id -> last_activity_timestamp
        self.blacklisted_ids = set()
        
    def register_activity(self, client_id: str):
        if client_id and client_id not in self.blacklisted_ids:
            self.active_clients[client_id] = time.time()
            
    def block(self, client_id: str):
        if client_id:
            self.blacklisted_ids.add(client_id)
            if client_id in self.active_clients:
                del self.active_clients[client_id]
            
    def unblock(self, client_id: str):
        if client_id:
            self.blacklisted_ids.discard(client_id)
            self.active_clients[client_id] = time.time()
        
    def get_clients_info(self):
        info = []
        # Сначала заблокированные
        for cid in sorted(self.blacklisted_ids):
            info.append({
                "client_id": cid,
                "last_activity": "Заблокирован",
                "status": "Заблокирован"
            })
        
        current_time = time.time()
        # Затем активные/офлайн
        for cid in sorted(self.active_clients.keys()):
            t = self.active_clients[cid]
            dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
            if current_time - t > 10.0:
                status = "Офлайн"
            else:
                status = "Активен"
            info.append({
                "client_id": cid,
                "last_activity": dt,
                "status": status
            })
        return info

blacklist_manager = BlacklistManager()

class ClientTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Извлекаем X-Client-ID из заголовков
        client_id = request.headers.get("X-Client-ID")
        
        # 2. Если заголовок отсутствует, проверяем также параметры формы или URL (на всякий случай)
        if not client_id:
            # Пытаемся взять из query параметров или form
            client_id = request.query_params.get("client_name") or request.query_params.get("client_id")
            
        if not client_id:
            client_id = "Неизвестный клиент"
            
        # 3. Проверяем черный список
        if client_id in blacklist_manager.blacklisted_ids:
            return JSONResponse(status_code=403, content={"detail": f"Forbidden: Client '{client_id}' is blocked."})
            
        # 4. Регистрируем активность
        if client_id != "Неизвестный клиент":
            blacklist_manager.register_activity(client_id)
            
        response = await call_next(request)
        return response

app = FastAPI(
    title="AI Contour API Server",
    description="REST API для автоматического оконтуривания критических органов на КТ",
    version="2.0.0"
)

app.add_middleware(ClientTrackerMiddleware)

# Инициализируем глобальный менеджер очереди
queue_manager = QueueManager(jobs_root="jobs")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "AI Contour API Server",
        "version": "2.0.0",
        "is_paused": queue_manager.is_paused
    }

@app.post("/api/jobs/upload")
async def upload_job(
    client_name: str = Form("Неизвестный клиент"),
    options_json: str = Form("{}"),
    file: UploadFile = File(...)
):
    """
    Загрузка ZIP-архива с DICOM-срезами и создание задачи в очереди.
    """
    logger.info(f"Получен запрос на сегментацию от '{client_name}' (Имя файла: {file.filename})")
    
    try:
        options = json.loads(options_json)
    except Exception as je:
        raise HTTPException(status_code=400, detail=f"Невалидный JSON в параметрах options_json: {je}")

    # Сохраняем загружаемый ZIP во временный файл
    try:
        temp_dir = Path(tempfile.gettempdir())
        temp_zip_path = temp_dir / f"uploaded_{uuid_str()}.zip"
        
        with open(temp_zip_path, "wb") as buffer:
            # Читаем чанками по 1 МБ для экономии ОЗУ при заливке 30-100 МБ
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
                
    except Exception as e:
        logger.error(f"Не удалось сохранить загруженный файл: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла на сервере: {e}")

    # Добавляем задачу в очередь менеджера
    try:
        job_id = queue_manager.add_job(
            client_name=client_name,
            temp_zip_path=temp_zip_path,
            options=options
        )
        
        job = queue_manager.get_job(job_id)
        pos = len(queue_manager.pending_queue)
        
        # Асинхронно воспроизводим короткий звуковой сигнал на сервере при новой задаче
        try:
            import winsound
            import threading
            def play_alert():
                try:
                    winsound.Beep(880, 100)
                    winsound.Beep(1109, 120)
                except Exception:
                    pass
            threading.Thread(target=play_alert, daemon=True).start()
        except Exception:
            pass

        return {
            "job_id": job_id,
            "status": "PENDING",
            "queue_position": pos,
            "message": f"Задача успешно добавлена в очередь. Позиция: {pos}"
        }
    except Exception as e:
        # В случае ошибки удаляем временный файл
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        logger.error(f"Ошибка при создании задачи: {e}")
        raise HTTPException(status_code=500, detail=f"Не удалось поставить задачу в очередь: {e}")

@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """
    Получение статуса выполнения и прогресса задачи.
    """
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача с указанным ID не найдена.")
        
    pos = "-"
    if job.status == "PENDING" and job.job_id in queue_manager.pending_queue:
        pos = queue_manager.pending_queue.index(job.job_id) + 1
        
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "patient_name": job.patient_name,
        "patient_id": job.patient_id,
        "queue_position": pos,
        "elapsed_seconds": round(job.elapsed_seconds, 1),
        "eta_seconds": round(job.eta_seconds, 1),
        "error_message": job.error_message,
        "is_server_paused": queue_manager.is_paused,
        "logs": job.logs
    }

@app.get("/api/jobs/{job_id}/download")
def download_job_result(job_id: str):
    """
    Скачивание ZIP-архива с результатом разметки RTSTRUCT.
    """
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача с указанным ID не найдена.")
        
    if job.status != "SUCCESS":
        raise HTTPException(
            status_code=400, 
            detail=f"Нельзя скачать результат задачи в статусе: {job.status}. Причина: {job.error_message or 'Не завершено'}"
        )
        
    result_path = Path(job.output_zip_path)
    if not result_path.exists():
        raise HTTPException(status_code=410, detail="Файл с результатами был удален на сервере (таймаут хранения).")
        
    return FileResponse(
        path=str(result_path),
        media_type="application/zip",
        filename=f"RS_AI_Contour_Result_{job_id[:8]}.zip"
    )

@app.delete("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, client_name: Optional[str] = None):
    """
    Отмена задачи с проверкой прав доступа.
    """
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача с указанным ID не найдена.")
        
    # Если передан client_name, это запрос от клиента
    if client_name is not None:
        if job.client_name != client_name:
            raise HTTPException(status_code=403, detail="Вы можете отменять только свои задачи.")
        if job.status == "SUCCESS":
            raise HTTPException(status_code=400, detail="Задача уже выполнена. Отмена невозможна.")

    success = queue_manager.cancel_job(job_id)
    if success:
        return {"job_id": job_id, "status": "CANCELLED", "message": "Задача успешно отменена."}
    else:
        return {
            "job_id": job_id,
            "status": job.status,
            "message": f"Невозможно отменить задачу в текущем статусе: {job.status}"
        }

@app.post("/api/jobs/{job_id}/prioritize")
def prioritize_job(job_id: str):
    """
    Поднятие задачи в начало очереди (приоритет).
    """
    with queue_manager.lock:
        if job_id in queue_manager.pending_queue:
            queue_manager.pending_queue.remove(job_id)
            queue_manager.pending_queue.insert(0, job_id)
            logger.info(f"Задача {job_id} перенесена в начало очереди (высокий приоритет) через API.")
            return {"job_id": job_id, "status": "PENDING", "message": "Задача перенесена в начало очереди."}
        else:
            job = queue_manager.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Задача с указанным ID не найдена.")
            return {
                "job_id": job_id,
                "status": job.status,
                "message": f"Невозможно изменить приоритет задачи в текущем статусе: {job.status}"
            }

@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """
    Возобновление упавшей или отмененной задачи.
    """
    success = queue_manager.resume_job(job_id)
    if success:
        return {"job_id": job_id, "status": "PENDING", "message": "Задача успешно возобновлена."}
    else:
        raise HTTPException(
            status_code=400, 
            detail="Не удалось возобновить задачу. Возможно, она не в статусе FAILED/CANCELLED или её файлы удалены."
        )

class ReorderPayload(BaseModel):
    job_ids: List[str]

@app.post("/api/queue/reorder")
def reorder_queue(payload: ReorderPayload):
    """
    Изменение порядка задач в очереди (только для PENDING задач).
    """
    success = queue_manager.reorder_queue(payload.job_ids)
    if success:
        return {"status": "success", "message": "Очередь успешно пересортирована."}
    else:
        raise HTTPException(status_code=400, detail="Не удалось пересортировать очередь.")

@app.get("/api/server/status")
def get_server_status():
    """
    Возвращает общие показатели состояния сервера и список задач.
    """
    return {
        "is_paused": queue_manager.is_paused,
        "pending_count": len(queue_manager.pending_queue),
        "jobs": queue_manager.get_queue_info()
    }

@app.post("/api/server/pause")
def pause_server():
    """
    Приостановить обработку очереди.
    """
    queue_manager.pause()
    return {"is_paused": True, "message": "Выполнение задач приостановлено."}

@app.post("/api/server/resume")
def resume_server():
    """
    Возобновить обработку очереди.
    """
    queue_manager.resume()
    return {"is_paused": False, "message": "Выполнение задач возобновлено."}

def uuid_str() -> str:
    import uuid
    return str(uuid.uuid4())

@app.get("/api/server/clients")
def get_clients():
    return blacklist_manager.get_clients_info()

@app.post("/api/server/clients/{client_id}/block")
def block_client(client_id: str):
    blacklist_manager.block(client_id)
    logger.info(f"Клиент '{client_id}' успешно заблокирован.")
    return {"status": "success", "message": f"Клиент '{client_id}' заблокирован."}

@app.post("/api/server/clients/{client_id}/unblock")
def unblock_client(client_id: str):
    blacklist_manager.unblock(client_id)
    logger.info(f"Клиент '{client_id}' успешно разблокирован.")
    return {"status": "success", "message": f"Клиент '{client_id}' разблокирован."}


# ================================================================================
# Эндпоинты для управления конфигурациями пресетов, цветов, локализаций и лицензий
# ================================================================================

class PresetPayload(BaseModel):
    name: str
    organs: List[str]
    colors: Optional[Dict[str, List[int]]] = None

class ColorsPayload(BaseModel):
    colors: Dict[str, List[int]]

class LicensePayload(BaseModel):
    license_key: str

@app.get("/api/config")
def get_server_config():
    """
    Возвращает полную конфигурацию пресетов, цветов, переводов и лицензий с сервера.
    """
    engine = queue_manager.engine
    return {
        "presets": engine.presets,
        "preset_colors": engine.preset_colors,
        "colors": engine.colors,
        "ru_names": engine.ru_names,
        "licenses": getattr(engine, "licenses", "")
    }

@app.post("/api/presets")
def save_server_preset(payload: PresetPayload):
    """
    Создает или обновляет OAR-пресет и его кастомные цвета на сервере.
    """
    engine = queue_manager.engine
    name = payload.name
    engine.presets[name] = payload.organs
    if payload.colors is not None:
        engine.preset_colors[name] = payload.colors
    else:
        engine.preset_colors.pop(name, None)
    
    engine.save_presets_config()
    logger.info(f"Пользовательский пресет '{name}' успешно сохранен на сервере.")
    return {"status": "success", "message": f"Пресет '{name}' успешно сохранен на сервере."}

@app.delete("/api/presets/{name}")
def delete_server_preset(name: str):
    """
    Удаляет OAR-пресет с сервера.
    """
    engine = queue_manager.engine
    if name in engine.presets:
        engine.presets.pop(name, None)
        engine.preset_colors.pop(name, None)
        engine.save_presets_config()
        logger.info(f"Пользовательский пресет '{name}' успешно удален с сервера.")
        return {"status": "success", "message": f"Пресет '{name}' успешно удален."}
    else:
        raise HTTPException(status_code=404, detail=f"Пресет '{name}' не найден на сервере.")

@app.post("/api/config/colors")
def save_server_colors(payload: ColorsPayload):
    """
    Обновляет глобальную цветовую схему органов на сервере.
    """
    engine = queue_manager.engine
    engine.colors.update(payload.colors)
    engine.save_presets_config()
    logger.info("Глобальные цвета органов успешно обновлены на сервере.")
    return {"status": "success", "message": "Цвета успешно сохранены на сервере."}

@app.post("/api/config/licenses")
def save_server_license(payload: LicensePayload):
    """
    Сохраняет или обновляет лицензионный ключ на сервере.
    """
    engine = queue_manager.engine
    engine.licenses = payload.license_key
    engine.save_presets_config()
    logger.info("Лицензионный ключ успешно обновлен на сервере.")
    return {"status": "success", "message": "Лицензионный ключ обновлен на сервере."}
