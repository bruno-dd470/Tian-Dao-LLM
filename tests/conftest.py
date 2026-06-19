"""
Configuration pytest pour le projet Tian-Dao-LLM.
Ajoute les chemins nécessaires pour les imports.
"""
import sys
import os

# Racine du projet (pour Endoregulated_AI_v27)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dossier demo/ (pour app.py)
DEMO_DIR = os.path.join(ROOT_DIR, "demo")

# Ajouter les deux chemins au sys.path
for path in [ROOT_DIR, DEMO_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)
