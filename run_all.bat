@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   MovieLens-1M RecSys Full Pipeline (Debug Mode)
echo ============================================================
echo.

REM Enter script directory
cd /d "%~dp0"

REM 1. Data validation
echo [1/5] Data validation...
python code\check.py
if errorlevel 1 (
    echo [ERROR] Data validation failed, please check data files
    pause
    exit /b 1
)
echo.

REM 2. Train basic dual-tower
echo [2/5] Train basic TwoTowerModel...
python code\train_deep.py
if errorlevel 1 (
    echo [ERROR] TwoTowerModel training failed
    pause
    exit /b 1
)
echo.

REM 3. Train SASRec dual-tower
echo [3/5] Train SASRec TwoTowerV2...
python code\train_v2.py
if errorlevel 1 (
    echo [ERROR] TwoTowerV2 training failed
    pause
    exit /b 1
)
echo.

REM 4. Train DIN reranking
echo [4/5] Train DIN reranking model...
python code\train_din.py
if errorlevel 1 (
    echo [ERROR] DIN training failed
    pause
    exit /b 1
)
echo.

REM 5. Full inference (recall + reranking + submit)
echo [5/5] Full inference pipeline...
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
