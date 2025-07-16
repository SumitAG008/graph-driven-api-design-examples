@echo off
echo ?? Installing Graph API Design Examples
echo ======================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ? Python 3 is required but not installed
    exit /b 1
)

echo ? Python found
for /f "tokens=*" %%i in ('python --version') do echo %%i

REM Test basic examples (no external dependencies)
echo ?? Testing basic examples...
cd examples\chapter_03_api_design
python working_examples.py

echo.
echo ? Basic examples work!
echo.
echo ?? Optional: Install full dependencies for API server:
echo    pip install -r requirements.txt
echo.
echo ?? Next steps:
echo    - Run examples: python examples\chapter_03_api_design\working_examples.py
echo    - Install deps:  pip install -r requirements.txt
echo    - Run tests:     python -m pytest tests/ -v
