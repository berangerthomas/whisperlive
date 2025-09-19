@echo off
setlocal

echo ###################################################################
echo #                                                                 #
echo #                 Lancement de WhisperLive                        #
echo #                                                                 #
echo ###################################################################
echo.

REM Verifie si l'environnement virtuel existe
if not exist .venv (
    echo ERREUR: L'environnement virtuel n'a pas ete trouve.
    echo Veuillez d'abord executer 'setup.bat' pour l'installer.
    pause
    exit /b 1
)

REM Activation de l'environnement virtuel
echo Activation de l'environnement virtuel...
call .\.venv\Scripts\activate

echo.
echo Que souhaitez-vous faire ?
echo 1. Lancer la transcription en direct depuis le microphone
echo 2. Transcrire un fichier audio (.wav)
echo.
choice /C 12 /M "Entrez votre choix (1 ou 2):"

if errorlevel 2 (
    goto transcribe_file
) else (
    goto live_transcription
)

:live_transcription
echo.
echo Lancement de la transcription en direct...
echo Appuyez sur Ctrl+C dans la fenetre du terminal pour arreter.
python whisperlive.py
goto end

:transcribe_file
echo.
set /p FILE_PATH="Veuillez glisser-deposer votre fichier .wav ici, puis appuyez sur Entree: "
REM Nettoyer le chemin si des guillemets ont ete ajoutes
set FILE_PATH=%FILE_PATH:"=%
python whisperlive.py --file "%FILE_PATH%"
goto end

:end
echo.
echo Le script est termine.
pause
