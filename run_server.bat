@echo off
:: ============================================================
::  Emotion Recognition v2 — Server Launcher
::  Double-click this file to start the server.
::  It finds ffmpeg automatically and sets PATH before Python starts.
:: ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo   Emotion Recognition v2 - Starting Server
echo ============================================================
echo.

:: ── Step 1: Find ffmpeg ──────────────────────────────────────────────────────
:: We search the most common locations. The first one that has ffmpeg.exe wins.

set FFMPEG_FOUND=0

:: Check if FFMPEG_BIN_DIR is already set by the user
if defined FFMPEG_BIN_DIR (
    if exist "%FFMPEG_BIN_DIR%\ffmpeg.exe" (
        echo [OK] Using FFMPEG_BIN_DIR: %FFMPEG_BIN_DIR%
        set PATH=%FFMPEG_BIN_DIR%;%PATH%
        set FFMPEG_FOUND=1
    )
)

:: Search Downloads folders for ALL users on this PC
if %FFMPEG_FOUND%==0 (
    for /d %%U in (C:\Users\*) do (
        for /d %%D in ("%%U\Downloads\ffmpeg*") do (
            if exist "%%D\bin\ffmpeg.exe" (
                echo [OK] Found ffmpeg: %%D\bin
                set PATH=%%D\bin;%PATH%
                set FFMPEG_FOUND=1
                goto :found
            )
            :: Some zips have a nested folder
            for /d %%E in ("%%D\*") do (
                if exist "%%E\bin\ffmpeg.exe" (
                    echo [OK] Found ffmpeg: %%E\bin
                    set PATH=%%E\bin;%PATH%
                    set FFMPEG_FOUND=1
                    goto :found
                )
            )
        )
    )
)

:: Check fixed common paths
if %FFMPEG_FOUND%==0 (
    for %%P in (
        "C:\ffmpeg\bin"
        "C:\Program Files\ffmpeg\bin"
        "C:\Program Files (x86)\ffmpeg\bin"
        "C:\tools\ffmpeg\bin"
    ) do (
        if exist "%%~P\ffmpeg.exe" (
            echo [OK] Found ffmpeg: %%~P
            set PATH=%%~P;%PATH%
            set FFMPEG_FOUND=1
            goto :found
        )
    )
)

:found
if %FFMPEG_FOUND%==0 (
    echo.
    echo [WARNING] ffmpeg.exe was NOT found automatically.
    echo.
    echo   Please edit run_server.bat and add your ffmpeg path, OR
    echo   set the FFMPEG_BIN_DIR variable before running:
    echo.
    echo     set FFMPEG_BIN_DIR=C:\path\to\ffmpeg\bin
    echo     run_server.bat
    echo.
    echo   Your ffmpeg is at:
    echo   C:\Users\Pragna\Downloads\ffmpeg-9.0-essentials_build\ffmpeg-9.0-essentials_build\bin\ffmpeg.exe
    echo   So run:
    echo     set FFMPEG_BIN_DIR=C:\Users\Pragna\Downloads\ffmpeg-9.0-essentials_build\ffmpeg-9.0-essentials_build\bin
    echo.
    pause
)

:: ── Step 2: Verify ffmpeg works ──────────────────────────────────────────────
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ffmpeg command failed even after PATH was set.
    echo         Check the path printed above is correct.
    pause
    exit /b 1
) else (
    echo [OK] ffmpeg is working.
)

:: ── Step 3: Start the server ─────────────────────────────────────────────────
echo.
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

uvicorn app:app --reload --port 8000

pause
