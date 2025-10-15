@echo off
echo ========================================
echo   KOLADA MCP - ChatGPT Integration
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Startar HTTP-server...
start "Kolada HTTP Server" powershell -NoExit -Command "$env:Path = 'C:\Users\Dennis Jernkrook\.local\bin;$env:Path'; Write-Host 'Startar Kolada HTTP Server...' -ForegroundColor Green; .\.venv\Scripts\python.exe run_http_server.py"

echo Vantar 8 sekunder for att servern ska starta...
timeout /t 8 /nobreak >nul

echo.
echo [2/2] Startar ngrok tunnel...
start "ngrok" powershell -NoExit -Command "Write-Host 'Startar ngrok...' -ForegroundColor Green; Write-Host 'Kopiera URL:en som borjar med https://' -ForegroundColor Yellow; Write-Host ''; ngrok http 8001"

echo.
echo ========================================
echo   Servrar startade!
echo ========================================
echo.
echo Tva PowerShell-fonstren oppnades:
echo   1. Kolada HTTP Server (port 8001)
echo   2. ngrok tunnel
echo.
echo VIKTIGT:
echo - Kopiera ngrok-URL:en fran ngrok-fonstret
echo - Anvand den i ChatGPT Actions
echo - Hall bada fonstren oppna medan du anvander ChatGPT
echo.
echo Tryck valfri tangent for att stanga detta fonster...
pause >nul
