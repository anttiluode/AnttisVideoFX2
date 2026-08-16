@echo off
setlocal
if "%LIVEPORTRAIT_HOME%"=="" (
  echo Set LIVEPORTRAIT_HOME to your official LivePortrait checkout first.
  echo Example: set LIVEPORTRAIT_HOME=E:\LivePortrait
  exit /b 1
)
if "%LIVEPORTRAIT_ENV%"=="" set LIVEPORTRAIT_ENV=LivePortrait
echo Starting LivePortrait bridge from %LIVEPORTRAIT_HOME% in conda env %LIVEPORTRAIT_ENV%
conda run --no-capture-output -n %LIVEPORTRAIT_ENV% python "%~dp0liveportrait_realtime_worker.py" --liveportrait-dir "%LIVEPORTRAIT_HOME%"
