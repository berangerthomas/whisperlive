@echo off
setlocal

echo ###################################################################
echo #                                                                 #
echo #      Configuration de l'environnement pour WhisperLive          #
echo #                                                                 #
echo ###################################################################
echo.

REM Verifie si Python est installe
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez l'installer depuis le Microsoft Store ou python.org et reessayer.
    pause
    exit /b 1
)

REM Creation de l'environnement virtuel
if not exist .venv (
    echo Creation de l'environnement virtuel...
    python -m venv .venv
)

echo Activation de l'environnement virtuel...
call .\.venv\Scripts\activate

echo.
echo ###################################################################
echo # Installation des dependances                                    #
echo ###################################################################
echo.
echo Installation de PyTorch (version CPU pour une compatibilite maximale)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installation des autres dependances depuis requirements.txt...
pip install -r requirements.txt

echo.
echo ###################################################################
echo # Configuration de l'acces Hugging Face                           #
echo ###################################################################
echo.
echo Le script va maintenant ouvrir deux pages dans votre navigateur :
echo 1. La page du modele de diarisation pour accepter les conditions.
echo 2. La page de vos tokens d'acces pour en creer/copier un.
echo.
pause
echo.

start "" "https://huggingface.co/pyannote/speaker-diarization-3.1"
start "" "https://huggingface.co/settings/tokens"

echo.
echo ETAPES A SUIVRE DANS VOTRE NAVIGATEUR :
echo -----------------------------------------
echo 1. Sur la premiere page, connectez-vous et cliquez sur :
echo    "Agree and access repository".
echo.
echo 2. Sur la deuxieme page, creez un nouveau token (role "read")
echo    et copiez-le.
echo -----------------------------------------
echo.

set /p HF_TOKEN="Une fois ces etapes terminees, collez votre token ici et appuyez sur Entree: "

echo HUGGING_FACE_TOKEN=%HF_TOKEN% > .env

echo.
echo Configuration terminee !
echo Vous pouvez maintenant lancer l'application en double-cliquant sur 'run.bat'.
echo.
pause
