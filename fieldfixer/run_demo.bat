@echo off
setlocal ENABLEDELAYEDEXPANSION

set SCRIPT_DIR=%~dp0
set IN_VIDEO=%SCRIPT_DIR%samples\alley_night\input.mp4
set BAKE_DIR=%SCRIPT_DIR%samples\alley_night\bake
set OUT_VIDEO=%SCRIPT_DIR%samples\alley_night\fixed.mp4
set CRF=18

set FIELD_FIXER_CMD=

where fieldfixer >nul 2>&1
if not errorlevel 1 (
    set FIELD_FIXER_CMD=fieldfixer
) else (
    for %%P in (python python3 py) do (
        where %%P >nul 2>&1
        if not errorlevel 1 (
            set FIELD_FIXER_CMD=%%P -m fieldfixer.cli
            goto :found_python
        )
    )
)

:found_python
if "%FIELD_FIXER_CMD%"=="" (
    echo [FieldFixer] Cannot find ^ieldfixer^, ^python^, ^python3^, or ^py^ on PATH.
    echo Install Python 3.11+, run ^"pip install -e .^", then rerun this script.
    pause
    exit /b 1
)

if not exist "%IN_VIDEO%" (
    echo [FieldFixer] Input video not found at "%IN_VIDEO%".
    echo Make sure samples\alley_night\input.mp4 exists before running the demo.
    pause
    exit /b 1
)

if not exist "%BAKE_DIR%" mkdir "%BAKE_DIR%"

echo === FieldFixer demo (identity sidecars) ===
echo CLI    : %FIELD_FIXER_CMD%
echo Input  : %IN_VIDEO%
echo Bake   : %BAKE_DIR%
echo Output : %OUT_VIDEO%

echo.
echo -> Baking...
%FIELD_FIXER_CMD% bake --in "%IN_VIDEO%" --out "%BAKE_DIR%" --modules rsnerf --modules deblurnerf --modules rawnerf --modules nerfw
if errorlevel 1 (
    echo Bake command failed.
    pause
    exit /b 1
)

echo.
echo -> Applying...
%FIELD_FIXER_CMD% apply --in "%IN_VIDEO%" --bake "%BAKE_DIR%" --out "%OUT_VIDEO%" --crf %CRF%
if errorlevel 1 (
    echo Apply command failed.
    pause
    exit /b 1
)

echo.
echo Demo complete. Output written to "%OUT_VIDEO%".
pause
