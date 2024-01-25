@echo off

rem Vérifier si pip est installé
where pip >nul 2>nul
if %errorlevel% neq 0 (
    echo "pip n'est pas installé. Veuillez installer Python avec pip avant d'exécuter ce script."
    exit /b 1
)

rem Installer les modules
pip install numpy pandas matplotlib scikit-learn tensorflow torch jupyter diffusers transformers accelerate ctransformers tk customtkinter sentencepiece

echo L'installation des modules Python pour le machine learning est terminée.