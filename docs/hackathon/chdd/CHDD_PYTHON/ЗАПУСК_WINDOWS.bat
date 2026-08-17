@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -c "import openpyxl" 2>nul
if errorlevel 1 py -m pip install -r requirements.txt
py "РАСЧЕТ_ЧДД.py"
if errorlevel 1 goto error
echo.
echo Результат находится в папке output.
pause
exit /b 0

:error
echo.
echo Проверьте сообщение об ошибке выше.
pause
exit /b 1
