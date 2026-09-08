@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   Amazon Reviews 2023 RecSys Full Pipeline
echo ============================================================
echo.

REM Enter script directory
cd /d "%~dp0"

REM 1. Data validation
echo [1/4] Data validation...
python code\check.py
if errorlevel 1 (
    echo [ERROR] Data validation failed, please check data files
    pause
    exit /b 1
)
echo.

REM 2. Train SASRec dual-tower
echo [2/4] Train SASRec TwoTowerV2...
python code\train_v2.py
if errorlevel 1 (
    echo [ERROR] TwoTowerV2 training failed
    pause
    exit /b 1
)
echo.

REM 3. Train Extended DIN reranking
echo [3/4] Train Extended DIN reranking model...
python code\train_din_ext.py
if errorlevel 1 (
    echo [ERROR] Extended DIN training failed
    pause
    exit /b 1
)
echo.

REM 4. Full inference (recall + reranking + submit)
echo [4/4] Full inference pipeline...
python code\inference_full.py
if errorlevel 1 (
    echo [ERROR] Inference failed
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   Full pipeline complete!
echo   Result: prediction_result\result_full_pipeline.csv
echo ============================================================
pause
