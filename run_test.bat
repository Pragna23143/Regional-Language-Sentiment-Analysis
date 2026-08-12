@echo off
:: ============================================================
::  Emotion Recognition v2 — Command-line Inference Launcher
::  Usage: run_test.bat --url "https://youtu.be/xxx"
::      or run_test.bat --file "C:\path\to\video.mp4"
:: ============================================================

setlocal enabledelayedexpansion

set FFMPEG_FOUND=0

if defined FFMPEG_BIN_DIR (
    if exist "%FFMPEG_BIN_DIR%\ffmpeg.exe" (
        set PATH=%FFMPEG_BIN_DIR%;%PATH%
        set FFMPEG_FOUND=1
    )
)

if %FFMPEG_FOUND%==0 (
    for /d %%U in (C:\Users\*) do (
        for /d %%D in ("%%U\Downloads\ffmpeg*") do (
            if exist "%%D\bin\ffmpeg.exe" (
                set PATH=%%D\bin;%PATH%
                set FFMPEG_FOUND=1
                goto :found
            )
            for /d %%E in ("%%D\*") do (
                if exist "%%E\bin\ffmpeg.exe" (
                    set PATH=%%E\bin;%PATH%
                    set FFMPEG_FOUND=1
                    goto :found
                )
            )
        )
    )
)

:found
python test.py %*
