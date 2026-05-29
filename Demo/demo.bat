@echo off
setlocal

cd /d "%~dp0.."
python "%~dp0demo.py" %*

echo.
pause
