@echo off
cd /d "%~dp0"
python ai_video_fx_causal.py --preset "Antti Causal Refresh"
if errorlevel 1 pause
