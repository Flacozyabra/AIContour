#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для сборки автономного Windows-клиента (.exe) для AI Contour.
Скрипт автоматически:
1. Устанавливает PyInstaller и Pillow во внутренний venv (если они не установлены).
2. Конвертирует png-иконку в ico-формат.
3. Компилирует клиент с полным исключением тяжелых ML библиотек (torch, totalsegmentator и т.д.).
4. Формирует готовую переносимую сборку в ZIP-архив с сохранением структуры настроек.
"""

import os
import sys
import subprocess
import shutil
import zipfile
from pathlib import Path

def print_banner(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")

def check_and_install_dependencies():
    print_banner("1. Проверка и установка сборочных зависимостей")
    
    # Определяем путь к pip в venv
    venv_pip = Path("venv") / "Scripts" / "pip.exe"
    if not venv_pip.exists():
        # Если venv/Scripts/pip.exe не найден, используем системный python/pip
        venv_pip = "pip"
        print("[WARNING] venv/Scripts/pip.exe не найден! Будет использован глобальный pip.")
    else:
        venv_pip = str(venv_pip)
        print(f"[INFO] Обнаружен pip в виртуальном окружении: {venv_pip}")
        
    try:
        # Проверяем pyinstaller
        import pyinstaller
        print("[OK] PyInstaller уже установлен.")
    except ImportError:
        print("[INFO] Установка PyInstaller...")
        subprocess.check_call([venv_pip, "install", "pyinstaller"])
        
    try:
        # Проверяем pillow (нужен для генерации иконки)
        from PIL import Image
        print("[OK] Pillow уже установлен.")
    except ImportError:
        print("[INFO] Установка Pillow для конвертации иконки...")
        subprocess.check_call([venv_pip, "install", "pillow"])

def generate_ico_icon():
    print_banner("2. Генерация иконки приложения")
    png_path = Path("app_icon.png")
    ico_path = Path("app_icon.ico")
    
    if not png_path.exists():
        print(f"[WARNING] Файл {png_path} не найден! Сборка будет выполнена со стандартной иконкой.")
        return False
        
    try:
        from PIL import Image
        print(f"[INFO] Конвертируем {png_path} в {ico_path}...")
        img = Image.open(png_path)
        
        # Windows ico поддерживает несколько разрешений
        sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        img.save(ico_path, format="ICO", sizes=sizes)
        print(f"[OK] Иконка {ico_path} успешно сгенерирована.")
        return True
    except Exception as e:
        print(f"[WARNING] Ошибка генерации иконки: {e}. Сборка продолжится со стандартной иконкой.")
        return False

def build_executable(has_icon):
    print_banner("3. Запуск компиляции через PyInstaller")
    
    # Путь к pyinstaller.exe в venv
    pyinstaller_bin = Path("venv") / "Scripts" / "pyinstaller.exe"
    if not pyinstaller_bin.exists():
        pyinstaller_bin = "pyinstaller"
        print("[WARNING] venv/Scripts/pyinstaller.exe не найден! Будет использован глобальный pyinstaller.")
    else:
        pyinstaller_bin = str(pyinstaller_bin)
        print(f"[INFO] Обнаружен PyInstaller в venv: {pyinstaller_bin}")
        
    # Формируем аргументы
    args = [
        pyinstaller_bin,
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name=AIContourClient",
    ]
    
    # Добавляем иконку, если создана
    if has_icon:
        args.append("--icon=app_icon.ico")
        
    # ИСКЛЮЧАЕМ ТЯЖЕЛЫЕ БИБЛИОТЕКИ СЕРВЕРА
    # Это ключевой момент, чтобы клиент весил 50МБ, а не 3ГБ!
    heavy_excludes = [
        "torch", "torchvision", "torchaudio", 
        "totalsegmentator", "SimpleITK", "nibabel", 
        "matplotlib", "pandas", "h5py", "scipy",
        "contour_engine" # движок тоже исключаем, клиент работает только по сети через API
    ]
    for m in heavy_excludes:
        args.append(f"--exclude-module={m}")
        
    # Основной файл запуска
    args.append("client_app.py")
    
    print(f"[INFO] Выполняется команда сборки:\n{' '.join(args)}")
    subprocess.check_call(args)
    print("[OK] Компиляция .exe успешно завершена.")

def package_portable_zip():
    print_banner("4. Подготовка переносимого (Portable) ZIP-дистрибутива")
    
    dist_dir = Path("dist")
    exe_file = dist_dir / "AIContourClient.exe"
    config_src = Path("config")
    
    if not exe_file.exists():
        raise RuntimeError(f"[ERROR] Собранный файл {exe_file} не найден!")
        
    # Создаем временную структуру для упаковки
    package_dir = dist_dir / "AIContourClient_Portable"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Копируем исполняемый файл
    print(f"[INFO] Копируем {exe_file.name} в портативный каталог...")
    shutil.copy2(exe_file, package_dir / exe_file.name)
    
    # 2. Копируем внешнюю папку настроек config/ (пользователь сможет редактировать пресеты/цвета)
    if config_src.exists() and config_src.is_dir():
        print("[INFO] Копируем конфигурационную папку config/...")
        shutil.copytree(config_src, package_dir / "config")
        
        # Удаляем временную статистику и логи, если они есть
        stats_file = package_dir / "config" / "statistics.json"
        if stats_file.exists():
            stats_file.unlink()
    else:
        print("[WARNING] Исходная папка config/ не найдена! Портативная сборка может быть неполной.")
        
    # 3. Создаем README
    readme_content = """=== AI Contour Client - Портативная версия ===

Инструкция по запуску и работе на компьютерах клиники:
1. Распакуйте содержимое архива в любую удобную папку на компьютере врача.
2. Запустите файл `AIContourClient.exe`.
3. Для работы программы НЕ ТРЕБУЕТСЯ установленный Python или мощная видеокарта.
4. ВНИМАНИЕ: Папка `config/` обязательно должна находиться рядом с файлом `AIContourClient.exe`.
   В этой папке хранятся анатомические пресеты, цвета контуров и переводы.
5. При первом запуске перейдите в раздел "Настройки" (иконка шестеренки сверху) 
   и укажите IP-адрес запущенного в клинике ИИ-сервера (например, http://192.168.1.100:8000).

Разработано для EvgeniyKrasnyanskiy/AIContur
"""
    readme_path = package_dir / "README_portable.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("[INFO] Создан файл README_portable.txt.")
        
    # 4. Упаковываем все в ZIP
    zip_filename = dist_dir / "AIContourClient_Portable.zip"
    if zip_filename.exists():
        zip_filename.unlink()
        
    print(f"[INFO] Упаковка в архив {zip_filename.name}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_f:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                # Сохраняем относительный путь внутри архива
                arcname = file_path.relative_to(package_dir)
                zip_f.write(file_path, arcname)
                
    # Очищаем временную папку
    shutil.rmtree(package_dir)
    
    print(f"[OK] Портативный ZIP-архив успешно создан: {zip_filename.resolve()}")
    print(f"Размер архива: {round(zip_filename.stat().st_size / (1024 * 1024), 2)} МБ")

def main():
    try:
        # Убедимся, что рабочая директория — это корень проекта
        os.chdir(Path(__file__).parent.resolve())
        
        check_and_install_dependencies()
        has_icon = generate_ico_icon()
        build_executable(has_icon)
        package_portable_zip()
        
        print_banner("СБОРКА УСПЕШНО ВЫПОЛНЕНА! ДИСТРИБУТИВ В ПАПКЕ dist/")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Произошел критический сбой при сборке: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
