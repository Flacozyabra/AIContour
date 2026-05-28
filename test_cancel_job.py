#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import subprocess
import unittest
from pathlib import Path
import psutil

# Добавим корень проекта в пути импорта
sys.path.append(str(Path(__file__).resolve().parent))

from server.queue_manager import QueueManager, ServerJob

class TestQueueManagerCancelJob(unittest.TestCase):
    def setUp(self):
        # Создаем временную директорию для тестов
        self.jobs_root = "test_jobs"
        self.qm = QueueManager(jobs_root=self.jobs_root)

    def tearDown(self):
        # Останавливаем фоновый воркер
        self.qm.stop_worker()
        # Очищаем временные директории
        import shutil
        if os.path.exists(self.jobs_root):
            shutil.rmtree(self.jobs_root, ignore_errors=True)

    def test_cancel_job_terminates_process_tree(self):
        # 1. Создаем фейковую задачу
        job_id = "test-job-uuid-12345"
        job = ServerJob(job_id, "TestClient", {"preset_name": "Все"})
        job.status = "PROCESSING"
        
        # 2. Запускаем цепочку процессов:
        # Родительский процесс запускает дочерний процесс, и оба засыпают.
        # Это идеально имитирует работу TotalSegmentator, запускающего CUDA-воркеры.
        code = (
            "import subprocess, time, sys;"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)']);"
            "time.sleep(15)"
        )
        parent_proc = subprocess.Popen([sys.executable, "-c", code])
        job.active_process = parent_proc
        
        # Сохраняем PID родителя
        parent_pid = parent_proc.pid
        
        # Даем немного времени процессам запуститься и породить потомка
        time.sleep(1.5)
        
        # Находим PID дочернего процесса
        try:
            psutil_parent = psutil.Process(parent_pid)
            children = psutil_parent.children(recursive=True)
            self.assertTrue(len(children) >= 1, "Дочерний процесс не был порожден!")
            child_pids = [c.pid for c in children]
        except psutil.NoSuchProcess:
            self.fail("Родительский процесс завершился до начала теста.")

        # Регистрируем задачу в менеджере
        self.qm.jobs[job_id] = job
        
        # 3. Вызываем отмену
        logger_name = "QueueManager"
        import logging
        logging.getLogger(logger_name).info("Тест: Вызываем отмену активной задачи.")
        success = self.qm.cancel_job(job_id)
        
        # Освобождаем ресурсы дескрипторов в Python Popen
        try:
            parent_proc.wait(timeout=1)
        except Exception:
            pass
        
        # 4. Проверяем успешность вызова
        self.assertTrue(success)
        self.assertEqual(job.status, "CANCELLED")

        
        # Даем процессам время на полное завершение (wait_procs завершается в cancel_job, но перестрахуемся)
        time.sleep(0.5)
        
        # 5. Проверяем, что ни один процесс из дерева больше не существует
        self.assertFalse(psutil.pid_exists(parent_pid), f"Родительский процесс {parent_pid} все еще жив!")
        for child_pid in child_pids:
            self.assertFalse(psutil.pid_exists(child_pid), f"Дочерний процесс {child_pid} все еще жив!")

if __name__ == "__main__":
    unittest.main()
