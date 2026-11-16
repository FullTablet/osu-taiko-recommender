@echo off
rem osu!taiko recommender installer: downloads a private copy of Python and the libraries the
rem app needs into this folder (nothing is installed or written elsewhere).
rem Run it again after unpacking a new app version over this folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\setup\install.ps1"
pause
