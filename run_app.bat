@echo off
cd /d "%~dp0"
echo ============================================
echo  Panel sweep analysis - starting
echo  Folder: %CD%
echo ============================================
echo.

if not exist "app.py" (
  echo ERROR: app.py was not found next to run_app.bat.
  echo Probably the ZIP was opened but not extracted.
  echo Right-click panel_app.zip, choose "Extract All", then run run_app.bat from the extracted folder.
  echo.
  pause
  exit /b
)

rem ---- find a Python that is 3.10 or newer (prefer 3.13 / 3.12) ----
set "PY="
for %%V in (3.13 3.12 3.14 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo ERROR: this app needs Python 3.10 or newer. The Python found here is:
  python --version
  echo.
  echo Please install Python 3.13 from https://www.python.org/downloads/
  echo In the installer tick "Add python.exe to PATH", then run run_app.bat again.
  echo The old Python can stay installed.
  echo.
  pause
  exit /b
)
echo Using: %PY%
%PY% --version

%PY% -c "import streamlit" 2>nul
if errorlevel 1 (
  echo Streamlit is not installed yet. Installing now, this takes 1-3 minutes...
  %PY% -m pip install -r requirements.txt
  %PY% -c "import streamlit" 2>nul
  if errorlevel 1 (
    echo.
    echo ERROR: the installation failed. Please send a screenshot of this window.
    pause
    exit /b
  )
)

rem Skip Streamlit's first-run e-mail question, which otherwise waits silently
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo Starting the app. A browser tab should open at http://localhost:8501
echo If it does not open, copy that address into the browser.
echo Keep this window open while using the app. Close it to stop the app.
echo.
%PY% -m streamlit run app.py --browser.gatherUsageStats false

echo.
echo The app stopped. If there is an error above, please send a screenshot.
pause
