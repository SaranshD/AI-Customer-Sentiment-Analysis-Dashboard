@echo off
setlocal enabledelayedexpansion

REM ===== CONFIG =====
set API_CMD=python main.py api
set API_URL=http://0.0.0.0:8000
set TEST_CMD=python test_api.py
set VENV_ACTIVATE=.venv\Scripts\activate

REM ===== LOG PREFIX =====
set LOG=[DEMO]

echo %LOG% Initializing demo environment...

REM ===== ACTIVATE VENV =====
call %VENV_ACTIVATE%
if errorlevel 1 (
    echo %LOG% Failed to activate virtual environment.
    exit /b 1
)

echo %LOG% Virtual environment activated.

REM ===== START API IN NEW WINDOW =====
echo %LOG% Starting API server...

start "API_SERVER" cmd /k ^
"prompt DemoAPI$G && call %VENV_ACTIVATE% && echo %LOG% API server starting... && %API_CMD%"

REM ===== WAIT FOR API TO BE READY =====
echo %LOG% Waiting for API to become available at %API_URL% ...

set MAX_RETRIES=20
set RETRY_DELAY=1
set COUNT=0

:wait_loop
set /a COUNT+=1

curl -s %API_URL% > nul 2>&1

if %errorlevel%==0 (
    echo %LOG% API is live.
    goto run_tests
)

if %COUNT% GEQ %MAX_RETRIES% (
    echo %LOG% API failed to start within expected time.
    goto cleanup
)

echo %LOG% Waiting... (%COUNT%/%MAX_RETRIES%)
timeout /t %RETRY_DELAY% > nul
goto wait_loop

:run_tests
echo %LOG% Running test script...

%TEST_CMD%
if errorlevel 1 (
    echo %LOG% Test script encountered an error.
) else (
    echo %LOG% Test script completed successfully.
)

:cleanup
echo %LOG% Shutting down API server...

REM Kill the API window by title
taskkill /FI "WINDOWTITLE eq API_SERVER*" /T /F > nul 2>&1

echo %LOG% Demo complete.

endlocal
exit /b 0