@echo off
chcp 65001 >nul 2>&1

echo.
echo === Установка Автоматического Генератора Глоссария ===
echo.

echo [1/8] Проверка Python...
python --version || (echo ОШИБКА: Python не найден && pause && exit /b 1)

echo [2/8] Создание venv...
if not exist "venv" python -m venv venv

echo [3/8] Активация venv...
call venv\Scripts\activate.bat

echo [4/8] Установка зависимостей...
pip install -r requirements.txt

echo [5/8] Загрузка spaCy...
python -m spacy download ru_core_news_sm

echo [6/8] Проверка Ollama...
where ollama || (echo ОШИБКА: Скачайте Ollama: https://ollama.com/download && pause && exit /b 1)

echo [7/8] Запуск сервера Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1 || (start "" /min ollama serve && timeout /t 5 /nobreak >nul)

echo [8/8] Загрузка модели...
if not exist "models" mkdir models
pip install huggingface-hub -q
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3-4B-Instruct-2507-GGUF', 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', local_dir='models')"
if errorlevel 1 (echo ОШИБКА: скачайте вручную https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF и положите в models\ && pause && exit /b 1)
(echo FROM models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) > Modelfile.tmp
ollama create qwen3:4b -f Modelfile.tmp
del Modelfile.tmp

:done
echo.
echo === УСТАНОВКА ЗАВЕРШЕНА ===
echo Запустите start.bat
pause