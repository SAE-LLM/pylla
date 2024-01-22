#!/bin/bash

# Installer les modules nécessaires pour le machine learning avec Python

# Vérifier si pip est installé
if command -v pip3 >/dev/null 2>&1; then
    echo "pip3 est déjà installé. Continuer l'installation des modules..."
else
    echo "pip3 n'est pas installé. Veuillez installer pip3 avant d'exécuter ce script."
    exit 1
fi

# Installer les modules
pip3 install numpy pandas matplotlib scikit-learn tensorflow torch jupyter diffusers transformers accelerate ctransformers

# Afficher un message de fin
echo "L'installation des modules Python pour le machine learning est terminée."
