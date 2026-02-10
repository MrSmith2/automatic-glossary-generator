:: start.bat
@echo off
chcp 65001 >nul 2>&1

if not exist "venv" (echo Сначала запустите install.bat && pause && exit /b 1)
call venv\Scripts\activate.bat

curl -s http://localhost:11434/api/tags >nul 2>&1 || (start "" /min ollama serve && timeout /t 5 /nobreak >nul)

streamlit run app/app.py --server.headless false --browser.gatherUsageStats false