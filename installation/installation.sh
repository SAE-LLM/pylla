#!/bin/bash

# Vérifier si le script est lancé avec sudo
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
    echo "Ce script peut nécessiter des droits sudo pour installer certains packages."
fi

# Installer les modules nécessaires pour le machine learning avec Python

# Vérifier si pip est installé
if command -v pip3 >/dev/null 2>&1; then
    echo "pip3 est déjà installé. Continuer l'installation des modules..."
else
    echo "pip3 n'est pas installé. Veuillez installer pip3 avant d'exécuter ce script."
    exit 1
fi

# Installer les modules
$SUDO pip3 install numpy pandas matplotlib scikit-learn tensorflow torch jupyter diffusers transformers accelerate ctransformers tk customtkinter sentencepiece Pillow

# Vérifier si apt-get est installé (pour Linux)
if command -v apt-get >/dev/null 2>&1; then
    echo "Installation de python-tk..."
    if ! $SUDO apt-get install -y python-tk; then
        echo "L'installation de python-tk a échoué. Veuillez vérifier les erreurs ci-dessus."
        exit 1
    fi
    echo "python-tk a été installé avec succès."
else
    echo "apt-get n'est pas installé. Veuillez installer apt-get manuellement pour continuer l'installation de python-tk."
    exit 1
fi

# Afficher un message de fin
echo "L'installation des modules Python pour le machine learning est terminée."
