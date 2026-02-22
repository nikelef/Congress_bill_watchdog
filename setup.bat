@echo off
setlocal

echo ============================================================
echo   US Congress Bill Monitor - Setup
echo ============================================================
echo.

REM Check for Python 3.8+
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)

if %PYMAJ% LSS 3 (
    echo ERROR: Python 3.8+ required. Found Python %PYVER%
    pause
    exit /b 1
)
if %PYMAJ% EQU 3 if %PYMIN% LSS 8 (
    echo ERROR: Python 3.8+ required. Found Python %PYVER%
    pause
    exit /b 1
)

echo [OK] Python %PYVER% detected
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo [OK] Dependencies installed
echo.

REM Create data directory
if not exist "data" mkdir data
echo [OK] data\ directory ready
echo.

REM Create default config.json if it doesn't exist
if not exist "config.json" (
    echo Creating default config.json...
    (
        echo {
        echo   "congress_api_key": "DEMO_KEY",
        echo   "check_interval_minutes": 60,
        echo   "bills_per_check": 50,
        echo   "congress": 119,
        echo   "track_recent_days": 3,
        echo   "keywords": [],
        echo   "email": {
        echo     "enabled": false,
        echo     "smtp_host": "smtp.gmail.com",
        echo     "smtp_port": 587,
        echo     "username": "",
        echo     "password": "",
        echo     "from": "",
        echo     "to": ""
        echo   }
        echo }
    ) > config.json
    echo [OK] config.json created
) else (
    echo [OK] config.json already exists - skipping
)
echo.

echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo Next steps:
echo.
echo  1. (Optional) Get a free Congress.gov API key:
echo       https://api.congress.gov/sign-up
echo     Then set "congress_api_key" in config.json
echo     (DEMO_KEY works but has lower rate limits)
echo.
echo  2. (Optional) Configure email in config.json:
echo       - Set "enabled": true
echo       - Fill in smtp_host, username, password, from, to
echo       - For Gmail: use an App Password, not your main password
echo         https://support.google.com/accounts/answer/185833
echo.
echo  3. Quick commands:
echo       python monitor.py --show        # One-time bill snapshot
echo       python monitor.py --test-email  # Test email config
echo       python monitor.py               # Start background daemon
echo       python monitor.py --reset       # Clear bill history
echo.
pause
