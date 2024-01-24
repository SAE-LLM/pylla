#!/bin/bash

# Installer les modules nécessaires pour le machine learning avec Python

# Vérifier si Python 3.7.9 est installé
if python3 --version 2>&1 | grep -q "3.7.9"; then
    echo "Python 3.7.9 est installé. Continuer l'installation des modules..."
else
    echo "Python 3.7.9 n'est pas installé. Veuillez installer Python 3.7.9 avec pip avant d'exécuter ce script."
    exit 1
fi

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
