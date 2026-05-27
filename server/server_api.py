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
from typing import Optional, List
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# Импортируем менеджер очереди
from server.queue_manager import QueueManager

logger = logging.getLogger("ServerAPI")

app = FastAPI(
    title="AI Contour API Server",
    description="REST API для автоматического оконтуривания критических органов на КТ",
    version="2.0.0"
)

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
