# Guide d'Installation et d'Utilisation pour Windows

Ce guide vous explique comment installer et lancer l'application WhisperLive sur un système d'exploitation Windows de manière simple et rapide.

## Prérequis

- **Python** : Assurez-vous que Python est installé sur votre machine. Si ce n'est pas le cas, vous pouvez le télécharger depuis le [Microsoft Store](https://www.microsoft.com/store/productId/9PJPW5LDXLZ5) ou le [site officiel python.org](https://www.python.org/downloads/windows/).

## Étape 1 : Installation (à faire une seule fois)

1.  **Lancez le script d'installation** : Double-cliquez sur le fichier `setup.bat`.
    
2.  **Patientez** : Une fenêtre de terminal va s'ouvrir et commencer à créer un environnement virtuel et à télécharger toutes les dépendances nécessaires. Cette étape peut prendre plusieurs minutes, en particulier pour le téléchargement de la bibliothèque PyTorch.
    
3.  **Configurez votre accès Hugging Face** :
    - Le script a besoin d'un token d'accès Hugging Face pour télécharger le modèle de diarisation (qui identifie qui parle). Ce modèle est protégé et nécessite deux actions de votre part.

    - **Action 1 : Accepter les conditions d'utilisation du modèle**
        - Avant même de créer un token, vous devez accepter les conditions d'utilisation du modèle de diarisation.
        - **Rendez-vous sur [cette page](https://huggingface.co/pyannote/speaker-diarization-3.1), connectez-vous, et cliquez sur "Agree and access repository" en haut de la page.**
        - Si vous ne faites pas cette étape, l'application ne pourra pas télécharger le modèle et affichera une erreur.

    - **Action 2 : Créer et fournir un token d'accès**
        - Une fois les conditions acceptées, allez dans vos paramètres Hugging Face : `Settings` > `Access Tokens`.
        - Créez un nouveau token (le rôle `read` est suffisant).
        - Copiez ce token.
        - Collez-le dans la fenêtre du terminal lorsque le script `setup.bat` vous le demande, puis appuyez sur `Entrée`.

4.  **Terminez l'installation** : Une fois le token sauvegardé, le script affichera "Configuration terminée !". Vous pouvez fermer la fenêtre.

## Étape 2 : Utilisation (à chaque lancement)

1.  **Lancez l'application** : Double-cliquez sur le fichier `run.bat`.

2.  **Choisissez le mode** : Le script vous demandera de choisir entre deux options :
    - **Option 1 : Transcription en direct**
        - Lance la transcription à partir de votre microphone.
        - Pour arrêter, fermez la fenêtre du terminal ou appuyez sur `Ctrl+C`.
    - **Option 2 : Transcrire un fichier**
        - Le script vous demandera de fournir le chemin d'un fichier audio.
        - Le plus simple est de **glisser-déposer votre fichier `.wav`** directement dans la fenêtre du terminal, puis d'appuyer sur `Entrée`.

3.  **Consultez les résultats** : La transcription sera affichée en direct dans le terminal et également sauvegardée dans un fichier `.txt` dans le même dossier.

Et voilà ! Votre application est prête à être utilisée.
