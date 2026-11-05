@echo off
cd /d "%~dp0"

if "%LICHESS_BOT_TOKEN%"=="" (
    echo LICHESS_BOT_TOKEN is not set.
    echo Set it once with: setx LICHESS_BOT_TOKEN "your_token_here"
    echo then open a new window and run this file again.
    pause
    exit /b 1
)

python "scripts\python\Lichess Bot\lichess_nn_bot.py"

echo.
echo The bot has stopped.
pause
